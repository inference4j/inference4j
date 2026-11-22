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
package io.github.inference4j.model;

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
import java.util.HexFormat;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;

/**
 * A local stand-in for the Hugging Face Hub, serving one repo: the revision API and file downloads. Files can
 * be changed between calls, which makes a new commit. Records every request path.
 */
final class FakeHub implements AutoCloseable {

    final String repoId;
    final List<String> requests = Collections.synchronizedList(new ArrayList<>());
    private final Map<String, byte[]> files = new LinkedHashMap<>();
    private final Set<String> lfsFiles;
    private final HttpServer server;
    private int commits;
    private String commit;
    /** When set, the revision API answers with this status instead. */
    volatile Integer revisionStatus;

    FakeHub(String repoId, Set<String> lfsFiles) throws IOException {
        this.repoId = repoId;
        this.lfsFiles = lfsFiles;
        this.server = HttpServer.create(new InetSocketAddress("127.0.0.1", 0), 0);
        server.createContext("/", this::handle);
        server.start();
        this.commit = nextCommit();
    }

    URI endpoint() {
        return URI.create("http://127.0.0.1:" + server.getAddress().getPort());
    }

    /** Sets a file's content; a new commit. */
    synchronized FakeHub put(String path, String content) {
        files.put(path, content.getBytes(StandardCharsets.UTF_8));
        commit = nextCommit();
        return this;
    }

    synchronized String commit() {
        return commit;
    }

    long count(String prefix) {
        synchronized (requests) {
            return requests.stream().filter(r -> r.startsWith(prefix)).count();
        }
    }

    long downloads(String path) {
        return count("/" + repoId + "/resolve/" + commit() + "/" + path);
    }

    long revisionChecks() {
        return count("/api/models/" + repoId + "/revision/main");
    }

    private String nextCommit() {
        return String.format("%040d", ++commits);
    }

    private synchronized void handle(HttpExchange exchange) throws IOException {
        String path = exchange.getRequestURI().getPath();
        requests.add(path);
        try (exchange) {
            if (path.equals("/api/models/" + repoId + "/revision/main")) {
                if (revisionStatus != null) {
                    send(exchange, revisionStatus, "{}".getBytes(StandardCharsets.UTF_8));
                    return;
                }
                send(exchange, 200, revisionJson().getBytes(StandardCharsets.UTF_8));
                return;
            }
            String prefix = "/" + repoId + "/resolve/";
            if (path.startsWith(prefix)) {
                String rest = path.substring(prefix.length());
                String file = rest.substring(rest.indexOf('/') + 1);
                byte[] content = files.get(file);
                if (content != null) {
                    send(exchange, 200, content);
                    return;
                }
            }
            send(exchange, 404, new byte[0]);
        }
    }

    private String revisionJson() {
        StringBuilder json = new StringBuilder("{\"sha\":\"" + commit + "\",\"siblings\":[");
        boolean first = true;
        for (Map.Entry<String, byte[]> file : files.entrySet()) {
            json.append(first ? "" : ",").append("{\"rfilename\":\"").append(file.getKey())
                    .append("\",\"blobId\":\"").append(gitBlobId(file.getValue())).append('"');
            if (lfsFiles.contains(file.getKey())) {
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
        server.stop(0);
    }
}
