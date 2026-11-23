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

// The inference4j dashboard. It reads the JSON API under ./api, which each framework integration serves:
//   GET    api/info                              {readOnly}
//   GET    api/cache                             CacheSummary
//   GET    api/cache/models                      [ModelView]
//   GET    api/cache/model?repo=&check=true      ModelView, after a Hub check
//   POST   api/cache/update?repo=                {repoId, status}
//   DELETE api/cache/model?repo=[&revision=]     EvictionResult
//   POST   api/cache/prune                       EvictionResult
(function () {
  'use strict';

  const UNITS = ['B', 'KB', 'MB', 'GB', 'TB', 'PB'];
  const TOP_N = 8;
  const POLL_IDLE_MS = 30000;
  const POLL_UPDATING_MS = 2000;

  const ICONS = {
    cache: '<svg viewBox="0 0 24 24"><ellipse cx="12" cy="6" rx="8" ry="3" fill="none" stroke="currentColor" stroke-width="1.8"/>' +
      '<path d="M4 6v6c0 1.7 3.6 3 8 3s8-1.3 8-3V6M4 12v6c0 1.7 3.6 3 8 3s8-1.3 8-3v-6" fill="none" stroke="currentColor" stroke-width="1.8"/></svg>'
  };

  function bytes(n) {
    if (n === null || n === undefined || isNaN(n)) {
      return '–';
    }
    let i = 0;
    let v = n;
    while (v >= 1000 && i < UNITS.length - 1) {
      v /= 1000;
      i++;
    }
    return (i === 0 ? v : v.toFixed(v < 10 ? 2 : v < 100 ? 1 : 0)) + ' ' + UNITS[i];
  }

  function ago(when) {
    const seconds = Math.round((Date.now() - new Date(when).getTime()) / 1000);
    if (seconds < 5) return 'just now';
    if (seconds < 60) return seconds + 's ago';
    const minutes = Math.round(seconds / 60);
    if (minutes < 60) return minutes + ' min ago';
    const hours = Math.round(minutes / 60);
    if (hours < 48) return hours + ' h ago';
    return Math.round(hours / 24) + ' days ago';
  }

  async function api(method, path, params) {
    const query = params ? '?' + new URLSearchParams(params).toString() : '';
    const headers = { 'Accept': 'application/json' };
    // Spring Security's cookie CSRF convention, when the application uses it
    const csrf = document.cookie.match(/(?:^|;\s*)XSRF-TOKEN=([^;]+)/);
    if (method !== 'GET' && csrf) {
      headers['X-XSRF-TOKEN'] = decodeURIComponent(csrf[1]);
    }
    const response = await fetch('api/' + path + query, {
      method: method,
      headers: headers,
      credentials: 'same-origin'
    });
    const text = await response.text();
    const body = text ? JSON.parse(text) : null;
    if (!response.ok) {
      const message = body && body.message ? body.message : response.status + ' ' + response.statusText;
      throw new Error(message);
    }
    return body;
  }

  document.addEventListener('alpine:init', function () {
    window.Alpine.data('dashboard', function () {
      return {
        sections: [{ id: 'cache', label: 'Cache', icon: ICONS.cache }],
        section: 'cache',
        readOnly: true,
        summary: {},
        models: [],
        expanded: {},
        busy: {},
        filter: '',
        sortKey: 'sizeBytes',
        sortDesc: true,
        loading: false,
        loadError: null,
        loadedAt: null,
        toasts: [],
        topN: TOP_N,
        timer: null,
        tick: 0,

        async init() {
          try {
            const info = await api('GET', 'info');
            this.readOnly = info.readOnly;
          } catch (e) {
            this.readOnly = true;
          }
          await this.load();
          // Re-render relative times
          setInterval(() => { this.tick++; }, 15000);
        },

        async load() {
          this.loading = true;
          try {
            const [summary, models] = await Promise.all([api('GET', 'cache'), api('GET', 'cache/models')]);
            // Keep the Hub check answers already shown
            const checks = {};
            this.models.forEach(m => { if (m.check) checks[m.repoId] = m.check; });
            models.forEach(m => {
              if (!m.check && checks[m.repoId] && checks[m.repoId].cachedRevision === m.revision) {
                m.check = checks[m.repoId];
              }
            });
            this.summary = summary;
            this.models = models;
            this.loadError = null;
            this.loadedAt = new Date().toISOString();
            this.$nextTick(() => this.renderCharts());
          } catch (e) {
            this.loadError = 'Cannot load the model cache: ' + e.message;
          } finally {
            this.loading = false;
            this.schedule();
          }
        },

        schedule() {
          clearTimeout(this.timer);
          const updating = this.summary.updating > 0 || this.models.some(m => m.updating);
          this.timer = setTimeout(() => this.load(), updating ? POLL_UPDATING_MS : POLL_IDLE_MS);
        },

        async check(m) {
          await this.run(m.repoId, async () => {
            const fresh = await api('GET', 'cache/model', { repo: m.repoId, check: 'true' });
            this.replace(fresh);
            this.expanded[m.repoId] = true;
            this.toast(fresh.check.updateAvailable
              ? m.repoId + ': ' + fresh.check.changedFiles.length + ' changed file(s) on the Hub'
              : m.repoId + ' is up to date', fresh.check.updateAvailable ? 'warn' : 'ok');
          });
        },

        async update(m) {
          await this.run(m.repoId, async () => {
            const result = await api('POST', 'cache/update', { repo: m.repoId });
            this.toast(result.status === 'already-running'
              ? m.repoId + ' is already updating' : 'Updating ' + m.repoId + '…', 'info');
            await this.load();
          });
        },

        async evict(m) {
          if (!confirm('Evict ' + m.repoId + '? Snapshots in use are kept; the model downloads again on its next load.')) {
            return;
          }
          await this.run(m.repoId, async () => {
            const result = await api('DELETE', 'cache/model', { repo: m.repoId });
            this.toast(this.evicted(result), 'ok');
            await this.load();
          });
        },

        async evictSnapshot(m, s) {
          if (!confirm('Delete snapshot ' + this.short(s.revision) + ' of ' + m.repoId + '?')) {
            return;
          }
          await this.run(m.repoId, async () => {
            const result = await api('DELETE', 'cache/model', { repo: m.repoId, revision: s.revision });
            this.toast(this.evicted(result), 'ok');
            await this.load();
          });
        },

        async prune() {
          await this.run('prune', async () => {
            const result = await api('POST', 'cache/prune');
            this.toast(this.evicted(result), 'ok');
            await this.load();
          });
        },

        async run(key, action) {
          this.busy[key] = true;
          try {
            await action();
          } catch (e) {
            this.toast(e.message, 'error');
          } finally {
            this.busy[key] = false;
          }
        },

        evicted(result) {
          let text = 'Freed ' + bytes(result.freedBytes) + ' (' + result.removed.length + ' snapshot'
            + (result.removed.length === 1 ? '' : 's') + ')';
          if (result.inUse.length) {
            text += '; ' + result.inUse.length + ' in use kept';
          }
          return text;
        },

        replace(fresh) {
          const i = this.models.findIndex(m => m.repoId === fresh.repoId);
          if (i >= 0) this.models[i] = fresh;
        },

        toast(text, kind) {
          const id = Date.now() + Math.random();
          this.toasts.push({ id: id, text: text, kind: kind || 'info' });
          setTimeout(() => this.dismiss(id), kind === 'error' ? 8000 : 4000);
        },

        dismiss(id) {
          this.toasts = this.toasts.filter(t => t.id !== id);
        },

        toggle(repoId) {
          this.expanded[repoId] = !this.expanded[repoId];
        },

        sortBy(key) {
          if (this.sortKey === key) {
            this.sortDesc = !this.sortDesc;
          } else {
            this.sortKey = key;
            this.sortDesc = key !== 'repoId';
          }
        },

        arrow(key) {
          return this.sortKey === key ? (this.sortDesc ? '↓' : '↑') : '';
        },

        visibleModels() {
          const f = this.filter.trim().toLowerCase();
          const key = this.sortKey;
          const list = this.models.filter(m => !f || m.repoId.toLowerCase().includes(f));
          list.sort((a, b) => {
            const x = a[key] ?? '';
            const y = b[key] ?? '';
            const order = x < y ? -1 : x > y ? 1 : 0;
            return this.sortDesc ? -order : order;
          });
          return list;
        },

        inUse(m) {
          return m.snapshots.reduce((n, s) => n + (s.leases > 0 ? 1 : 0), 0);
        },

        otherUsed() {
          const d = this.summary.disk;
          return d ? Math.max(0, d.totalBytes - d.usableBytes - this.summary.sizeBytes) : 0;
        },

        renderCharts() {
          // A container can still be zero wide before layout; draw once it has a size, and again when it changes
          ['disk-chart', 'size-chart'].forEach(id => {
            const el = document.getElementById(id);
            if (el && !el._observed && window.ResizeObserver) {
              el._observed = true;
              let width = el.clientWidth;
              new ResizeObserver(() => {
                if (el.clientWidth !== width) {
                  width = el.clientWidth;
                  this.renderCharts();
                }
              }).observe(el);
            }
          });
          const d = this.summary.disk;
          const disk = document.getElementById('disk-chart');
          if (d && disk) {
            // A cache of a few GB on a large disk would be a hairline: draw it at 2% at least; the legend and
            // tooltips keep the real sizes
            const minimum = d.totalBytes * 0.02;
            const cache = this.summary.sizeBytes > 0 ? Math.max(this.summary.sizeBytes, minimum) : 0;
            const slices = [
              { value: cache, size: this.summary.sizeBytes, className: 'slice-cache', meta: 'inference4j cache' },
              { value: this.otherUsed(), size: this.otherUsed(), className: 'slice-other', meta: 'Other files' },
              { value: d.usableBytes, size: d.usableBytes, className: 'slice-free', meta: 'Free' }
            ];
            new window.Chartist.PieChart(disk, {
              series: slices.map(s => ({ value: s.value, className: s.className, meta: s.meta + ': ' + bytes(s.size) }))
            }, { donut: true, donutWidth: 22, showLabel: false, startAngle: 0 })
              .on('draw', ctx => {
                if (ctx.type === 'slice') {
                  ctx.element.elem('title').text(ctx.meta);
                }
              });
          }

          const sizes = document.getElementById('size-chart');
          const top = [...this.models].sort((a, b) => b.sizeBytes - a.sizeBytes).slice(0, TOP_N);
          if (sizes && !top.length) {
            sizes.innerHTML = '';
            sizes.style.height = '0';
          }
          if (sizes && top.length) {
            sizes.style.height = (top.length * 34 + 40) + 'px';
            new window.Chartist.BarChart(sizes, {
              labels: top.map(m => this.name(m.repoId)),
              series: [top.map(m => ({ value: m.sizeBytes, meta: m.repoId }))]
            }, {
              horizontalBars: true,
              reverseData: true,
              seriesBarDistance: 0,
              chartPadding: { top: 0, right: 56, bottom: 0, left: 0 },
              axisX: { labelInterpolationFnc: v => bytes(v), onlyInteger: true, scaleMinSpace: 70 },
              axisY: { offset: 200, showGrid: false,
                labelInterpolationFnc: v => v.length > 30 ? v.slice(0, 29) + '…' : v }
            }).on('draw', ctx => {
              if (ctx.type === 'bar') {
                ctx.element.elem('title').text(ctx.meta + ': ' + bytes(ctx.value.x));
              }
            });
          }
        },

        bytes: bytes,

        ago(when) {
          // Read tick so relative times re-render
          return this.tick >= 0 ? ago(when) : '';
        },

        name(repoId) {
          return repoId.slice(repoId.indexOf('/') + 1);
        },

        short(revision) {
          return revision ? revision.slice(0, 8) : '–';
        },

        percent(part, whole) {
          if (!whole) return '–';
          const p = 100 * part / whole;
          return (p < 1 && p > 0 ? p.toFixed(2) : p.toFixed(p < 10 ? 1 : 0)) + '%';
        },

        share(part, whole) {
          return whole ? Math.max(1, Math.round(100 * part / whole)) : 0;
        }
      };
    });
  });
})();
