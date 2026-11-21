---
layout: archive
title: "Publications"
permalink: /publications/
author_profile: true
---

{% if author.googlescholar %}
  You can also find my articles on <u><a href="{{author.googlescholar}}">my Google Scholar profile</a>.</u>
{% endif %}

{% include base_path %}

<div class="collection-entries collection-entries--publications">
{% assign publications = site.publications | sort: "publication_order" %}
{% for post in publications %}
  {% include collection-entry.html post=post kind="publication" %}
{% endfor %}
</div>
