(function () {
  var input = document.getElementById('blog-search-input');
  var status = document.getElementById('blog-search-status');
  var defaultPosts = document.getElementById('default-posts');
  var resultsWrap = document.getElementById('search-results');
  if (!input) return;

  var indexPromise = null;
  var debounceTimer = null;

  function loadIndex() {
    if (!indexPromise) {
      indexPromise = fetch(baseUrlJoin('/search.json')).then(function (res) {
        return res.json();
      });
    }
    return indexPromise;
  }

  function baseUrlJoin(path) {
    var base = window.SITE_BASEURL || '';
    return (base + path).replace(/\/{2,}/g, '/');
  }

  function escapeHtml(str) {
    return String(str)
      .replace(/&/g, '&amp;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;')
      .replace(/"/g, '&quot;');
  }

  function renderCard(post) {
    var imageHtml = post.image
      ? '<a href="' + post.url + '"><img class="rounded mb-4" src="' + post.image + '" alt="' + escapeHtml(post.title) + '"></a>'
      : '';
    return (
      '<div class="col-md-6 mb-5">' +
        '<div class="card">' +
          imageHtml +
          '<div class="card-block">' +
            '<h2 class="card-title h4 serif-font"><a href="' + post.url + '">' + escapeHtml(post.title) + '</a></h2>' +
            '<p class="card-text text-muted">' + escapeHtml(post.excerpt) + '</p>' +
            '<div class="metafooter">' +
              '<div class="wrapfooter small d-flex align-items-center">' +
                '<span class="author-meta">By <span class="post-name">' + escapeHtml(post.author) + ', </span> on <span class="post-date">' + escapeHtml(post.date) + '</span></span>' +
                '<div class="clearfix"></div>' +
              '</div>' +
            '</div>' +
          '</div>' +
        '</div>' +
      '</div>'
    );
  }

  // Matches a query term against a haystack, tolerating simple English
  // plurals (e.g. "competition" <-> "competitions") since posts are
  // often titled in the singular while people search in the plural.
  function termMatches(haystack, term) {
    if (haystack.indexOf(term) !== -1) return true;
    if (term.length > 3) {
      if (term.charAt(term.length - 1) === 's' && haystack.indexOf(term.slice(0, -1)) !== -1) return true;
      if (term.charAt(term.length - 1) !== 's' && haystack.indexOf(term + 's') !== -1) return true;
    }
    return false;
  }

  function scorePost(post, terms, includeContent) {
    var title = post.title.toLowerCase();
    var meta = (post.excerpt + ' ' + (post.categories || []).join(' ')).toLowerCase();
    var content = includeContent ? (post.content || '').toLowerCase() : '';
    var matchedTerms = 0;
    var score = 0;
    terms.forEach(function (term) {
      if (termMatches(title, term)) { score += 3; matchedTerms++; }
      else if (termMatches(meta, term)) { score += 2; matchedTerms++; }
      else if (includeContent && termMatches(content, term)) { score += 1; matchedTerms++; }
    });
    return matchedTerms === terms.length ? score : 0;
  }

  // Every query term must match somewhere for a post to qualify (so
  // generic words like "data" or "science" alone don't return the whole
  // blog). We first try title/excerpt/categories only; if that finds
  // nothing we widen the search into the full post body, which is what
  // catches things like a library name ("imblearn") that's only
  // mentioned in a code sample rather than in the title or excerpt.
  function findMatches(posts, terms) {
    var narrow = posts
      .map(function (post) { return { post: post, score: scorePost(post, terms, false) }; })
      .filter(function (m) { return m.score > 0; });

    var pool = narrow.length ? narrow : posts
      .map(function (post) { return { post: post, score: scorePost(post, terms, true) }; })
      .filter(function (m) { return m.score > 0; });

    return pool
      .sort(function (a, b) { return b.score - a.score; })
      .map(function (m) { return m.post; });
  }

  function runSearch(query) {
    query = query.trim().toLowerCase();

    if (!query) {
      resultsWrap.hidden = true;
      resultsWrap.innerHTML = '';
      defaultPosts.hidden = false;
      status.hidden = true;
      return;
    }

    defaultPosts.hidden = true;
    resultsWrap.hidden = false;

    loadIndex().then(function (posts) {
      var terms = query.split(/\s+/).filter(Boolean);
      var matches = findMatches(posts, terms);

      status.hidden = false;
      status.textContent = matches.length
        ? matches.length + ' result' + (matches.length === 1 ? '' : 's') + ' for "' + query + '"'
        : 'No results for "' + query + '"';

      resultsWrap.innerHTML = matches.map(renderCard).join('');
    });
  }

  input.addEventListener('input', function () {
    clearTimeout(debounceTimer);
    var value = input.value;
    debounceTimer = setTimeout(function () { runSearch(value); }, 200);
  });

  var form = document.getElementById('blog-search-form');
  if (form) {
    form.addEventListener('submit', function (e) {
      e.preventDefault();
      clearTimeout(debounceTimer);
      runSearch(input.value);
    });
  }
})();
