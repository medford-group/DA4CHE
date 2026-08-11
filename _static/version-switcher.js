/*
 * version-switcher.js -- "other editions" control for the published book.
 *
 * Reads the published version index and renders a selector into the left sidebar.
 * Registered via `sphinx: config: html_js_files` in _config.yml, so it ships inside
 * every build -- including frozen editions, which is the point:
 *
 *   - INDEX_URL is ABSOLUTE, not relative to the page. An edition published in
 *     August cannot know about one published the following January, so it has to ask
 *     the live index rather than carry a snapshot of it.
 *   - Every failure path is silent. The index does not exist on the development site
 *     or in a local `_build/html`, and a book that logged errors or rendered an empty
 *     control in those cases would be worse than one that simply omits it.
 *
 * A frozen edition keeps whatever version of this file it shipped with, so changes
 * here reach past editions only when they are next patched. Keep it self-contained.
 */
(function () {
  "use strict";

  var INDEX_URL = "/DA4CHE/versions.json";
  var EDITION_RE = /\/DA4CHE\/(v\d{4}\.\d{2})\//;

  if (typeof window.fetch !== "function") return;

  var CSS =
    ".da4che-versions{padding:.5rem .25rem;margin-bottom:.5rem;font-size:.85rem}" +
    ".da4che-versions label{display:block;margin-bottom:.25rem;font-weight:600;" +
    "text-transform:uppercase;letter-spacing:.04em;font-size:.7rem;opacity:.75;" +
    "color:var(--pst-color-text-muted,#646464)}" +
    ".da4che-versions select{width:100%;padding:.3rem .4rem;border-radius:.25rem;" +
    "border:1px solid var(--pst-color-border,#c9c9c9);" +
    "background:var(--pst-color-background,#fff);" +
    "color:var(--pst-color-text-base,#000);font-size:.85rem}" +
    ".da4che-versions .da4che-versions__note{margin-top:.3rem;font-size:.72rem;" +
    "line-height:1.3;color:var(--pst-color-text-muted,#646464)}";

  /* Label an entry the way a reader would name it: edition, title, and the patch
     level when there is one. Patch 0 is left unsaid -- most editions never move. */
  function labelFor(entry) {
    /* Editions are named by their identifier; the development entry has no meaningful
       one, so it is named by its title alone. */
    var isEdition = /^v\d{4}\.\d{2}$/.test(entry.id || "");
    var label = isEdition ? entry.id : "";
    if (entry.title) label += (label ? " — " : "") + entry.title;
    if (!label) label = entry.id || "unknown";
    if (entry.patch) label += " (patch " + entry.patch + ")";
    if (entry.browsable === false) label += " — download only";
    return label;
  }

  function currentEditionId() {
    var m = EDITION_RE.exec(window.location.pathname);
    return m ? m[1] : null;
  }

  /* Try to land on the same page in the target edition, falling back to its front
     page. Chapters get added, renamed and moved between editions, so the same path
     is a good guess but never a guarantee -- and a 404 is a worse answer than the
     table of contents. */
  function navigate(target, tailPath) {
    if (!tailPath) {
      window.location.href = target;
      return;
    }
    var candidate = target.replace(/\/$/, "") + "/" + tailPath;
    fetch(candidate, { method: "HEAD" })
      .then(function (res) {
        window.location.href = res && res.ok ? candidate : target;
      })
      .catch(function () {
        window.location.href = target;
      });
  }

  function mount(node) {
    var host =
      document.querySelector(".sidebar-primary-items__start") ||
      document.querySelector(".bd-sidebar-primary") ||
      document.querySelector(".bd-article-container") ||
      document.getElementById("main-content");
    if (!host) return false;
    host.insertBefore(node, host.firstChild);
    return true;
  }

  function render(data) {
    var entries = (data && data.versions) || [];
    if (data && data.development) entries = entries.concat([data.development]);
    /* One entry means there is nothing to switch between. */
    if (entries.length < 2) return;

    var here = currentEditionId();
    var m = EDITION_RE.exec(window.location.pathname);
    var tailPath = m
      ? window.location.pathname.slice(m.index + m[0].length)
      : null;

    var style = document.createElement("style");
    style.textContent = CSS;
    document.head.appendChild(style);

    var wrap = document.createElement("div");
    wrap.className = "da4che-versions";

    var label = document.createElement("label");
    label.setAttribute("for", "da4che-version-select");
    label.textContent = "Edition";
    wrap.appendChild(label);

    var select = document.createElement("select");
    select.id = "da4che-version-select";
    var matched = false;

    entries.forEach(function (entry) {
      if (!entry || !entry.url) return;
      var opt = document.createElement("option");
      opt.value = entry.url;
      opt.textContent = labelFor(entry);
      if (entry.id && entry.id === here) {
        opt.selected = true;
        matched = true;
      }
      select.appendChild(opt);
    });

    if (!select.options.length) return;

    /* Served from somewhere the index does not describe -- a local build, or the
       development site. Say so rather than implying one of the editions is current. */
    if (!matched) {
      var opt = document.createElement("option");
      opt.value = "";
      opt.textContent = "This copy (unreleased)";
      opt.selected = true;
      select.insertBefore(opt, select.firstChild);
    }

    select.addEventListener("change", function () {
      if (select.value) navigate(select.value, tailPath);
    });
    wrap.appendChild(select);

    if (data && data.index_url) {
      var note = document.createElement("div");
      note.className = "da4che-versions__note";
      var a = document.createElement("a");
      a.href = data.index_url;
      a.textContent = "All editions";
      note.appendChild(a);
      wrap.appendChild(note);
    }

    mount(wrap);
  }

  function load() {
    fetch(INDEX_URL, { cache: "no-cache" })
      .then(function (res) {
        if (!res.ok) throw new Error("unavailable");
        return res.json();
      })
      .then(render)
      .catch(function () {
        /* No index reachable: leave the page exactly as it was. */
      });
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", load);
  } else {
    load();
  }
})();
