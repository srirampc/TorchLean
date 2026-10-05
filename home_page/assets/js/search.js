(function () {
  var root = document.documentElement;
  var themeButton = document.querySelector("[data-theme-toggle]");
  var themeColor = document.querySelector('meta[name="theme-color"]');

  function storedTheme() {
    try {
      return window.localStorage.getItem("torchlean-theme");
    } catch (error) {
      return null;
    }
  }

  function saveTheme(theme) {
    try {
      window.localStorage.setItem("torchlean-theme", theme);
    } catch (error) {
      // The toggle should still work when localStorage is unavailable.
    }
  }

  function setTheme(theme) {
    root.setAttribute("data-theme", theme);
    if (themeColor) {
      themeColor.setAttribute("content", theme === "dark" ? "#101418" : "#ffffff");
    }
    if (themeButton) {
      themeButton.setAttribute("aria-label", theme === "dark" ? "Use light theme" : "Use dark theme");
    }
  }

  var initialTheme = storedTheme();
  if (initialTheme === "dark" || initialTheme === "light") {
    setTheme(initialTheme);
  } else {
    setTheme("light");
  }

  if (themeButton) {
    themeButton.addEventListener("click", function () {
      var nextTheme = root.getAttribute("data-theme") === "dark" ? "light" : "dark";
      setTheme(nextTheme);
      saveTheme(nextTheme);
    });
  }

  var openButton = document.querySelector("[data-search-open]");
  var overlay = document.querySelector("[data-search-overlay]");
  var closeButton = document.querySelector("[data-search-close]");
  var input = document.querySelector("[data-search-input]");
  var results = document.querySelector("[data-search-results]");
  var modeButtons = Array.prototype.slice.call(document.querySelectorAll("[data-search-mode]"));
  var routeLink = document.querySelector("[data-search-route]");

  if (!openButton || !overlay || !closeButton || !input || !results) {
    return;
  }

  var index = [];
  var loaded = false;
  var loadFailed = false;
  var activeMode = "site";
  var previousFocus = null;

  function siteRoot() {
    var scripts = document.getElementsByTagName("script");
    for (var i = scripts.length - 1; i >= 0; i -= 1) {
      var src = scripts[i].getAttribute("src") || "";
      var marker = "/assets/js/search.js";
      var at = src.indexOf(marker);
      if (at >= 0) {
        return src.slice(0, at);
      }
    }
    return "";
  }

  function loadIndex() {
    if (loaded) {
      return Promise.resolve(index);
    }
    return fetch(siteRoot() + "/search-index.json")
      .then(function (response) {
        if (!response.ok) {
          throw new Error("search index unavailable");
        }
        return response.json();
      })
      .then(function (items) {
        index = items;
        loaded = true;
        loadFailed = false;
        return index;
      })
      .catch(function () {
        loadFailed = true;
        return [];
      });
  }

  function normalize(value) {
    return String(value || "").toLowerCase();
  }

  function escapeHtml(value) {
    return String(value || "")
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;");
  }

  function docsSearchUrl(query) {
    var trimmed = String(query || "").trim();
    var url = siteRoot() + "/docs/search.html";
    return trimmed ? url + "?q=" + encodeURIComponent(trimmed) : url;
  }

  function updateRouteLink(query) {
    if (!routeLink) {
      return;
    }
    routeLink.setAttribute("href", docsSearchUrl(query));
    routeLink.textContent = query.trim() ? 'Search API declarations for "' + query.trim() + '"' : "Search API declarations";
  }

  function setMode(mode) {
    activeMode = mode === "api" ? "api" : "site";
    modeButtons.forEach(function (button) {
      var selected = button.getAttribute("data-search-mode") === activeMode;
      button.classList.toggle("active", selected);
      button.setAttribute("aria-pressed", selected ? "true" : "false");
    });
    input.setAttribute(
      "placeholder",
      activeMode === "api" ? "Search definitions, theorems, modules..." : "Search site pages..."
    );
    render(index, input.value);
  }

  function score(item, terms) {
    var title = normalize(item.title);
    var section = normalize(item.section);
    var text = normalize(item.text);
    var total = 0;

    terms.forEach(function (term) {
      if (title.indexOf(term) >= 0) {
        total += 8;
      }
      if (section.indexOf(term) >= 0) {
        total += 4;
      }
      if (text.indexOf(term) >= 0) {
        total += 2;
      }
    });

    return total;
  }

  function render(items, query) {
    updateRouteLink(query);

    if (activeMode === "api") {
      results.innerHTML =
        '<p class="search-empty">Search DocGen declarations: definitions, theorems, classes, structures, and modules.</p>' +
        '<a class="search-result search-result-action" href="' + escapeHtml(docsSearchUrl(query)) + '">' +
        '<span>API declarations</span>' +
        '<strong>' + (query.trim() ? 'Search "' + escapeHtml(query.trim()) + '"' : "Open declaration search") + '</strong>' +
        '<em>Open the generated API search page with this query.</em>' +
        '</a>';
      return;
    }

    if (loadFailed) {
      results.innerHTML = '<p class="search-empty">Search is unavailable in this build.</p>';
      return;
    }

    if (!query.trim()) {
      results.innerHTML = '<p class="search-empty">Type to search guide pages, examples, verification notes, CUDA material, and updates. Use API declarations for Lean names.</p>';
      return;
    }

    var terms = normalize(query).split(/\s+/).filter(Boolean);
    var ranked = items
      .map(function (item) {
        return { item: item, score: score(item, terms) };
      })
      .filter(function (entry) {
        return entry.score > 0;
      })
      .sort(function (a, b) {
        return b.score - a.score || a.item.title.localeCompare(b.item.title);
      })
      .slice(0, 8);

    if (ranked.length === 0) {
      results.innerHTML =
        '<p class="search-empty">No site results for "' + escapeHtml(query) + '". Try API declarations for Lean names.</p>';
      return;
    }

    results.innerHTML = ranked
      .map(function (entry) {
        var item = entry.item;
        return (
          '<a class="search-result" href="' + escapeHtml(item.url) + '">' +
          '<span>' + escapeHtml(item.section) + '</span>' +
          '<strong>' + escapeHtml(item.title) + '</strong>' +
          '<em>' + escapeHtml(item.text) + '</em>' +
          '</a>'
        );
      })
      .join("");
  }

  function openSearch() {
    if (overlay.hidden) {
      previousFocus = document.activeElement;
    }
    overlay.hidden = false;
    document.body.classList.add("search-active");
    input.focus();
    input.select();
    loadIndex().then(function (items) {
      if (!overlay.hidden) {
        render(items, input.value);
      }
    });
  }

  function closeSearch() {
    overlay.hidden = true;
    document.body.classList.remove("search-active");
    if (previousFocus && previousFocus.isConnected) {
      previousFocus.focus();
    } else {
      openButton.focus();
    }
  }

  openButton.addEventListener("click", openSearch);
  closeButton.addEventListener("click", closeSearch);
  overlay.addEventListener("click", function (event) {
    if (event.target === overlay) {
      closeSearch();
    }
  });
  input.addEventListener("input", function () {
    render(index, input.value);
  });
  input.addEventListener("keydown", function (event) {
    if (event.key === "Enter" && activeMode === "api") {
      event.preventDefault();
      window.location.href = docsSearchUrl(input.value);
    }
  });
  modeButtons.forEach(function (button) {
    button.addEventListener("click", function () {
      setMode(button.getAttribute("data-search-mode"));
      input.focus();
    });
  });
  document.addEventListener("keydown", function (event) {
    if (event.key === "Tab" && !overlay.hidden) {
      var controls = overlay.querySelectorAll('button, input, a[href]');
      var first = controls[0];
      var last = controls[controls.length - 1];
      if (event.shiftKey && document.activeElement === first) {
        event.preventDefault();
        last.focus();
      } else if (!event.shiftKey && document.activeElement === last) {
        event.preventDefault();
        first.focus();
      }
    }
    if ((event.metaKey || event.ctrlKey) && event.key.toLowerCase() === "k") {
      event.preventDefault();
      openSearch();
    }
    if (event.key === "Escape" && !overlay.hidden) {
      closeSearch();
    }
  });
})();
