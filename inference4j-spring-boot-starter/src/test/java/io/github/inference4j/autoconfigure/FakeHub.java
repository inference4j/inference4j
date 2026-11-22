/*
 * Copyright 2026 the original author or authors.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      https://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package io.github.inference4j.autoconfigure;

import com.sun.net.httpserver.HttpExchange;
import com.sun.net.httpserver.HttpServer;

import java.io.IOException;
import java.io.OutputStream;
import java.net.InetSocketAddress;
import java.net.URI;
import java.nio.charset.StandardCharsets;
import java.security.MessageDigest;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.HexFormat;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;

/**
 * A copy of the core tests' {@code FakeHub}, for the starter's tests.
 *
 * <p>A local stand-in for the Hugging Face Hub: the revision API and file downloads, for one or more repos. Each
 * change makes a new commit, and downloads at an older commit serve that commit's content, with the commit in an
 * {@code X-Repo-Commit} header. Records every request path; downloads can be failed or held at a gate.
 */
final class FakeHub implements AutoCloseable {

    /** The repo the single-repo methods act on. */
    final String repoId;
    final List<String> requests = Collections.synchronizedList(new ArrayList<>());
    private final Map<String, Repo> repos = new LinkedHashMap<>();
    private final Set<String> failing = ConcurrentHashMap.newKeySet();
    private final HttpServer server;
    private final ExecutorService executor = Executors.newCachedThreadPool();
    private int commits;
    /** When set, the revision API answers with this status instead. */
    volatile Integer revisionStatus;
    /** Whether downloads carry the {@code X-Repo-Commit} header. */
    volatile boolean commitHeader = true;
    private volatile Gate gate;

    private static final class Repo {
        final Set<String> lfsFiles;
        final Map<String, Map<String, byte[]>> history = new HashMap<>();
        Map<String, byte[]> files = new LinkedHashMap<>();
        String commit;

        Repo(Set<String> lfsFiles) {
            this.lfsFiles = lfsFiles;
        }
    }

    /** Holds downloads until released. */
    static final class Gate {
        private final CountDownLatch waiting = new CountDownLatch(1);
        private final CountDownLatch open = new CountDownLatch(1);

        void awaitWaiting() throws InterruptedException {
            if (!waiting.await(10, TimeUnit.SECONDS)) {
                throw new IllegalStateException("No download reached the gate");
            }
        }

        void release() {
            open.countDown();
        }

        private void pass() {
            waiting.countDown();
            try {
                open.await(30, TimeUnit.SECONDS);
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
            }
        }
    }

    FakeHub(String repoId, Set<String> lfsFiles) throws IOException {
        this.repoId = repoId;
        addRepo(repoId, lfsFiles);
        this.server = HttpServer.create(new InetSocketAddress("127.0.0.1", 0), 0);
        server.createContext("/", this::handle);
        server.setExecutor(executor);
        server.start();
    }

    URI endpoint() {
        return URI.create("http://127.0.0.1:" + server.getAddress().getPort());
    }

    /** Serves another repo too. */
    synchronized FakeHub addRepo(String id, Set<String> lfsFiles) {
        Repo repo = new Repo(lfsFiles);
        repos.put(id, repo);
        commit(repo);
        return this;
    }

    /** Sets a file's content; a new commit. */
    FakeHub put(String path, String content) {
        return put(repoId, path, content.getBytes(StandardCharsets.UTF_8));
    }

    FakeHub put(String repo, String path, String content) {
        return put(repo, path, content.getBytes(StandardCharsets.UTF_8));
    }

    synchronized FakeHub put(String repo, String path, byte[] content) {
        Repo r = repos.get(repo);
        r.files = new LinkedHashMap<>(r.files);
        r.files.put(path, content);
        commit(r);
        return this;
    }

    /** Deletes a file; a new commit. */
    synchronized FakeHub delete(String path) {
        Repo r = repos.get(repoId);
        r.files = new LinkedHashMap<>(r.files);
        r.files.remove(path);
        commit(r);
        return this;
    }

    synchronized String commit() {
        return commit(repoId);
    }

    synchronized String commit(String repo) {
        return repos.get(repo).commit;
    }

    /** Makes downloads of {@code path} fail with HTTP 500, or succeed again. */
    void failDownload(String path, boolean fail) {
        if (fail) {
            failing.add(path);
        } else {
            failing.remove(path);
        }
    }

    /** Holds the next downloads until the returned gate is released. */
    Gate holdDownloads() {
        Gate g = new Gate();
        gate = g;
        return g;
    }

    long count(String prefix) {
        synchronized (requests) {
            return requests.stream().filter(r -> r.startsWith(prefix)).count();
        }
    }

    /** Downloads of {@code path} from the default repo, at any revision. */
    long downloads(String path) {
        return downloads(repoId, path);
    }

    long downloads(String repo, String path) {
        String prefix = "/" + repo + "/resolve/";
        synchronized (requests) {
            return requests.stream().filter(r -> r.startsWith(prefix)
                    && r.substring(prefix.length()).substring(r.substring(prefix.length()).indexOf('/') + 1)
                    .equals(path)).count();
        }
    }

    /** Downloads of {@code path} from the default repo at {@code revision}. */
    long downloadsAt(String revision, String path) {
        String request = "/" + repoId + "/resolve/" + revision + "/" + path;
        synchronized (requests) {
            return requests.stream().filter(request::equals).count();
        }
    }

    long revisionChecks() {
        return revisionChecks(repoId);
    }

    long revisionChecks(String repo) {
        return count("/api/models/" + repo + "/revision/main");
    }

    private void commit(Repo repo) {
        repo.commit = String.format("%040d", ++commits);
        repo.history.put(repo.commit, repo.files);
    }

    private void handle(HttpExchange exchange) throws IOException {
        String path = exchange.getRequestURI().getPath();
        requests.add(path);
        try (exchange) {
            for (String id : repoIds()) {
                if (path.equals("/api/models/" + id + "/revision/main")) {
                    Integer status = revisionStatus;
                    if (status != null) {
                        send(exchange, status, "{}".getBytes(StandardCharsets.UTF_8));
                    } else {
                        send(exchange, 200, revisionJson(id).getBytes(StandardCharsets.UTF_8));
                    }
                    return;
                }
                String prefix = "/" + id + "/resolve/";
                if (path.startsWith(prefix)) {
                    serveFile(exchange, id, path.substring(prefix.length()));
                    return;
                }
            }
            send(exchange, 404, new byte[0]);
        }
    }

    private void serveFile(HttpExchange exchange, String id, String rest) throws IOException {
        String revision = rest.substring(0, rest.indexOf('/'));
        String file = rest.substring(rest.indexOf('/') + 1);
        byte[] content;
        String commit;
        synchronized (this) {
            Repo repo = repos.get(id);
            commit = revision.equals("main") ? repo.commit : revision;
            Map<String, byte[]> files = repo.history.get(commit);
            content = files != null ? files.get(file) : null;
        }
        Gate g = gate;
        if (g != null) {
            g.pass();
        }
        if (failing.contains(file)) {
            send(exchange, 500, new byte[0]);
            return;
        }
        if (content == null) {
            send(exchange, 404, new byte[0]);
            return;
        }
        if (commitHeader) {
            exchange.getResponseHeaders().add("X-Repo-Commit", commit);
        }
        send(exchange, 200, content);
    }

    private synchronized List<String> repoIds() {
        return new ArrayList<>(repos.keySet());
    }

    private synchronized String revisionJson(String id) {
        Repo repo = repos.get(id);
        StringBuilder json = new StringBuilder("{\"sha\":\"" + repo.commit + "\",\"siblings\":[");
        boolean first = true;
        for (Map.Entry<String, byte[]> file : repo.files.entrySet()) {
            json.append(first ? "" : ",").append("{\"rfilename\":\"").append(file.getKey())
                    .append("\",\"blobId\":\"").append(gitBlobId(file.getValue())).append('"');
            if (repo.lfsFiles.contains(file.getKey())) {
                json.append(",\"lfs\":{\"sha256\":\"").append(hex("SHA-256", file.getValue())).append("\"}");
            }
            json.append('}');
            first = false;
        }
        return json.append("]}").toString();
    }

    private static String gitBlobId(byte[] content) {
        byte[] header = ("blob " + content.length + "\0").getBytes(StandardCharsets.US_ASCII);
        byte[] all = new byte[header.length + content.length];
        System.arraycopy(header, 0, all, 0, header.length);
        System.arraycopy(content, 0, all, header.length, content.length);
        return hex("SHA-1", all);
    }

    private static String hex(String algorithm, byte[] bytes) {
        try {
            return HexFormat.of().formatHex(MessageDigest.getInstance(algorithm).digest(bytes));
        } catch (Exception e) {
            throw new IllegalStateException(e);
        }
    }

    private static void send(HttpExchange exchange, int status, byte[] body) throws IOException {
        exchange.sendResponseHeaders(status, body.length == 0 ? -1 : body.length);
        if (body.length > 0) {
            try (OutputStream out = exchange.getResponseBody()) {
                out.write(body);
            }
        }
    }

    @Override
    public void close() {
        Gate g = gate;
        if (g != null) {
            g.release();
        }
        server.stop(0);
        executor.shutdownNow();
    }
}
