require 'jekyll'
require 'nokogiri'
site = Jekyll::Site.new(Jekyll.configuration('source' => Dir.pwd, 'destination' => '/private/tmp/academic-entry-test-output', 'quiet' => true, 'url' => 'https://example.test', 'baseurl' => '/sub'))
source = File.read('_includes/collection-entry.html')
render = lambda do |post, kind|
  output = Liquid::Template.parse(source).render!({'include' => {'post' => post, 'kind' => kind}, 'base_path' => 'https://example.test/sub'}, registers: {site: site})
  Nokogiri::HTML.fragment(output)
end
check = lambda { |condition, message| raise message unless condition }
# Real Liquid and Jekyll filters, independent fixtures: check user-visible output.
minimal = render.call({'title' => 'Minimal note', 'url' => '/notes/minimal/'}, 'note')
check.call(minimal.at_css('h2 a')['href'] == 'https://example.test/sub/notes/minimal/', 'Local note link lost base path')
check.call(minimal.at_css('.collection-entry__links').nil?, 'Minimal note has empty resource row')
check.call(minimal.at_css('.collection-entry__meta').nil?, 'Minimal note has empty metadata')
rich = render.call({'title' => 'Note with **math**', 'url' => '/notes/math/', 'date' => Time.utc(2025,1,2), 'excerpt' => '<p>A short <em>summary</em> & useful context.</p>', 'header' => {'teaser' => 'note.png'}}, 'note')
check.call(rich.at_css('h2 strong').text == 'math', 'Title formatting lost')
check.call(rich.at_css('.collection-entry__excerpt').text == 'A short summary & useful context.', 'Note summary not converted to safe plain text')
check.call(rich.at_css('img')['src'] == 'https://example.test/sub/images/note.png', 'Local thumbnail has incorrect path')
check.call(rich.at_css('time').text == 'Jan 2025', 'Note date is incorrect')
long = render.call({'title' => 'Long note', 'url' => '/notes/long/', 'excerpt' => ('word ' * 90)}, 'note')
check.call(long.at_css('.collection-entry__excerpt').text.split.size <= 36, 'Note summary is not compact')
pub = render.call({'title' => 'Research result', 'url' => '/publication/result/', 'collection' => 'Masters Thesis', 'link' => 'https://example.org/paper?a=1&b=2', 'paperurl' => 'https://example.org/result.pdf', 'citation' => 'Original citation.', 'abstract' => 'An **abstract**.'}, 'publication')
check.call(pub.at_css('h2 a')['href'] == 'https://example.org/paper?a=1&b=2', 'External publication link altered')
check.call(pub.css('.collection-entry__links a').map(&:text).map(&:strip) == ['Paper', 'PDF'], 'Resource labels missing or incorrect')
check.call(pub.css('details').size == 2 && pub.css('details[open]').empty?, 'Citation and abstract must be collapsed initially')
check.call(pub.css('details').text.include?('Original citation.'), 'Original citation lost')
empty = render.call({'title' => 'No resources', 'url' => '/publication/minimal/'}, 'publication')
check.call(empty.at_css('.collection-entry__links').nil?, 'Publication has empty resources row')
puts 'PASS: note links, optional fields, formatted titles, safe excerpts, short summaries, thumbnails, dates, publication resources, citations, and abstract disclosure.'

relative = render.call({'title' => 'Linked note', 'url' => '/notes/local/', 'link' => '/source/'}, 'note')
check.call(relative.at_css('h2 a')['href'] == 'https://example.test/sub/source/', 'Relative title URL lost the site base path')
check.call(relative.at_css('.collection-entry__read')['href'] == 'https://example.test/sub/notes/local/', 'Read note must keep the local permalink')
external = render.call({'title' => 'External title', 'url' => '/notes/local/', 'link' => 'https://other.test/article'}, 'note')
check.call(external.at_css('h2 a')['href'] == 'https://other.test/article', 'External title URL changed')
check.call(external.at_css('.collection-entry__read')['href'] == 'https://example.test/sub/notes/local/', 'External title must not hide the local note')
blank = render.call({'title' => 'Blank fields', 'url' => '/notes/blank/', 'excerpt' => '  ', 'date' => ''}, 'note')
check.call(blank.at_css('.collection-entry__excerpt').nil?, 'Blank excerpt created an empty paragraph')
blank_pub = render.call({'title' => 'Blank publication', 'url' => '/publication/blank/', 'link' => '', 'paperurl' => '  ', 'type' => '', 'venue' => '', 'date' => '', 'citation' => '', 'abstract' => ' '}, 'publication')
check.call(blank_pub.at_css('.collection-entry__links').nil?, 'Blank resources created an empty control')
check.call(blank_pub.at_css('.collection-entry__meta').nil?, 'Blank metadata created an empty row')
check.call(blank_pub.at_css('details').nil?, 'Blank citation or abstract created an empty disclosure')
relative_pub = render.call({'title' => 'Local PDF', 'url' => '/publication/local/', 'paperurl' => '/files/paper.pdf'}, 'publication')
check.call(relative_pub.at_css('.collection-entry__links a')['href'] == 'https://example.test/sub/files/paper.pdf', 'Relative PDF lost its base path')
puts 'PASS: base-path resource links, local note permalinks, and blank optional values.'

['/source/?ref=doi:10.1234/test', 'source/?ref=doi:10.1234/test'].each do |path|
  note = render.call({'title' => 'DOI reference', 'url' => '/notes/reference/', 'link' => path}, 'note')
  check.call(note.at_css('h2 a')['href'] == 'https://example.test/sub/source/?ref=doi:10.1234/test', 'Colon in query string must not drop the base path')
end
puts 'PASS: colons in local URL query strings retain the deployment prefix.'

satcom = render.call({'title' => 'SatcomLLM', 'url' => '/publication/satcomllm/', 'code' => 'https://github.com/esa-sceva', 'huggingface' => 'https://huggingface.co/esa-sceva'}, 'publication')
check.call(satcom.css('.collection-entry__links a').map { |a| [a.text.strip, a['href']] } == [['Code', 'https://github.com/esa-sceva'], ['Hugging Face', 'https://huggingface.co/esa-sceva']], 'Code and Hugging Face resources must both be visible')
puts 'PASS: Code and Hugging Face publication resources.'

undated = render.call({'title' => 'Under-review submission', 'url' => '/publication/undated/', 'venue' => 'Under review in a journal', 'date' => Time.utc(2026,11,21), 'hide_date' => true}, 'publication')
check.call(undated.at_css('time').nil?, 'An undated submission must hide Jekyll’s inferred file date')
puts 'PASS: undated submissions do not display an inferred publication year.'

slides_url = 'https://www.dropbox.com/scl/fi/example/slides.pdf?rlkey=test&dl=0'
talk = render.call({'title' => 'Research Overview: Work done at IIT Hyderabad', 'url' => '/talks/research-overview/', 'link' => slides_url, 'slidesurl' => slides_url, 'venue' => 'IIT Hyderabad', 'hide_date' => true, 'date' => Time.utc(2026,11,21)}, 'talk')
check.call(talk.at_css('article')['class'].split.include?('collection-entry--note'), 'Talk must reuse the note-card style')
check.call(talk.at_css('.collection-entry__read').text.include?('Read presentation'), 'Talk must use a presentation action label')
check.call(talk.at_css('.collection-entry__read')['href'] == 'https://example.test/sub/talks/research-overview/', 'Talk details link must retain the base path')
check.call(talk.at_css('.collection-entry__slides')['href'] == slides_url, 'Talk must include its exact slides PDF URL')
check.call(talk.at_css('h2 a')['href'] == slides_url, 'Research overview title must link to the PDF')
check.call(talk.at_css('time').nil?, 'Undated talk must not display an inferred date')
puts 'PASS: talk cards, presentation labels, PDF resources, and hidden inferred dates.'

publication_placeholder = render.call({'title' => 'Publication awaiting a figure', 'url' => '/publication/figure/'}, 'publication')
check.call(publication_placeholder.at_css('.collection-entry__thumbnail--placeholder'), 'Publication without a figure must show an image placeholder')
publication_figure = render.call({'title' => 'Publication with figure', 'url' => '/publication/figure/', 'header' => {'teaser' => 'paper-figure.png'}}, 'publication')
check.call(publication_figure.at_css('img')['src'] == 'https://example.test/sub/images/paper-figure.png', 'Publication figure must use its configured teaser')
check.call(publication_figure.at_css('.collection-entry__thumbnail--placeholder').nil?, 'Real publication figure must replace the placeholder')
check.call(minimal.at_css('.collection-entry__thumbnail--placeholder').nil?, 'Notes must not gain publication placeholders')
puts 'PASS: publication figure placeholders and configured image replacement.'
