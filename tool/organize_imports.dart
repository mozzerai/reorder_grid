// Organizes the top-of-file directive block in every Dart file under
// `reorder_grid/lib/`. Groups imports and exports into three sections separated by
// blank lines, sorted alphabetically within each group:
//
//   1. dart:        SDK imports
//   2. package:     third-party package imports
//   3. package:reorder_grid  own package imports
//
// Order: imports block, blank line, exports block, blank line, parts block.
// Preserves any leading comment header (license/copyright) above the first
// directive. Preserves multi-line `show`/`hide`/`as` clauses and inline
// comments attached to individual directives.
//
// Run from the package root:
//   dart run tool/organize_imports.dart

import 'dart:io';

import 'package:path/path.dart' as p;

const String _ownPackagePrefix = 'package:reorder_grid/';

// Matches one directive (`import`, `export`, or `part`) starting at a line
// boundary, captures all chars up to and including the terminating `;`. URIs
// never contain `;`, and combinator clauses only contain identifiers/commas,
// so `[^;]+;` is safe.
final RegExp _directiveRe = RegExp(
  r'''(^|\n)([ \t]*)(import|export|part)\s+([^;]+);''',
);

// URI inside a directive body.
final RegExp _uriRe = RegExp(r'''(['"])([^'"]+)\1''');

class _Directive {
  final String keyword; // import | export | part
  final String uri;
  final String
  fullText; // full directive text including any leading comment lines

  _Directive(this.keyword, this.uri, this.fullText);

  int get bucket {
    if (uri.startsWith('dart:')) return 0;
    if (uri.startsWith(_ownPackagePrefix)) return 2;
    if (uri.startsWith('package:')) return 1;
    return 3; // relative / other — kept untouched at end
  }
}

Future<void> main(List<String> args) async {
  final libDir = Directory(p.join(Directory.current.path, 'lib'));
  if (!libDir.existsSync()) {
    stderr.writeln('Error: lib/ not found. Run from the package root.');
    exit(1);
  }

  var filesScanned = 0;
  var filesChanged = 0;

  final files = libDir
      .listSync(recursive: true)
      .whereType<File>()
      .where((f) => f.path.endsWith('.dart'))
      .toList();

  for (final file in files) {
    filesScanned++;
    final original = await file.readAsString();
    final updated = _organize(original);
    if (updated != null && updated != original) {
      await file.writeAsString(updated);
      filesChanged++;
    }
  }

  stdout.writeln('Scanned: $filesScanned files');
  stdout.writeln('Changed: $filesChanged files');
}

String? _organize(String content) {
  // Find directive block: from first directive match to end of last
  // contiguous directive (allowing blank lines / leading comments between
  // them).
  final matches = _directiveRe.allMatches(content).toList();
  if (matches.isEmpty) return null;

  // Block spans from the first character that is part of the first directive
  // (or its attached leading comments) to the end of the last directive
  // before any non-directive code appears.
  //
  // Strategy: collect a contiguous run of directives where the gap between
  // them only contains whitespace and `//`/`/* ... */` comments.

  final directives = <_Directive>[];
  var blockStart = matches.first.start;
  var blockEnd = matches.first.end;

  // Adjust blockStart to skip the leading `\n` captured in group(1).
  if (content[blockStart] == '\n') blockStart++;

  // Walk matches, attach attached leading comments to each directive.
  int prevEnd = blockStart;
  for (final m in matches) {
    final directiveTextStart = m.start + (content[m.start] == '\n' ? 1 : 0);
    final gap = content.substring(prevEnd, directiveTextStart);

    // If the gap contains non-comment / non-whitespace text, this directive
    // is no longer part of the contiguous block — stop.
    if (!_isOnlyCommentsAndWhitespace(gap)) {
      break;
    }

    final keyword = m.group(3)!;
    final body = m.group(4)!;
    final uriMatch = _uriRe.firstMatch(body);
    final uri = uriMatch?.group(2) ?? '';

    final fullText = content.substring(
      m.start + (content[m.start] == '\n' ? 1 : 0),
      m.end,
    );

    directives.add(_Directive(keyword, uri, fullText));

    blockEnd = m.end;
    prevEnd = m.end;
  }

  if (directives.isEmpty) return null;

  // Skip trailing newline after last directive — it remains as the separator
  // before the rest of the file.
  final rest = content.substring(blockEnd);

  // Header: everything before the block start.
  final header = content.substring(0, blockStart);

  // Group directives by kind and bucket.
  final importsBy = <int, List<_Directive>>{};
  final exportsBy = <int, List<_Directive>>{};
  final parts = <_Directive>[];

  for (final d in directives) {
    if (d.keyword == 'part') {
      parts.add(d);
    } else if (d.keyword == 'import') {
      importsBy.putIfAbsent(d.bucket, () => []).add(d);
    } else {
      exportsBy.putIfAbsent(d.bucket, () => []).add(d);
    }
  }

  final out = StringBuffer();
  out.write(header);

  _writeGroup(out, importsBy);
  if (exportsBy.isNotEmpty) {
    if (out.isNotEmpty && !out.toString().endsWith('\n\n')) {
      out.write('\n');
    }
    _writeGroup(out, exportsBy);
  }
  if (parts.isNotEmpty) {
    if (out.isNotEmpty && !out.toString().endsWith('\n\n')) {
      out.write('\n');
    }
    for (final p in parts) {
      out.write(p.fullText);
      if (!p.fullText.endsWith('\n')) out.write('\n');
    }
  }

  // Ensure exactly one blank line between directive block and rest.
  var tail = rest;
  while (tail.startsWith('\n')) {
    tail = tail.substring(1);
  }
  if (tail.isNotEmpty) {
    out.write('\n');
    out.write(tail);
  }

  return out.toString();
}

void _writeGroup(StringBuffer out, Map<int, List<_Directive>> by) {
  final ordered = [0, 1, 2, 3];
  var firstGroup = true;
  for (final bucket in ordered) {
    final list = by[bucket];
    if (list == null || list.isEmpty) continue;
    list.sort((a, b) => a.uri.compareTo(b.uri));
    if (!firstGroup) out.write('\n');
    for (final d in list) {
      out.write(d.fullText);
      if (!d.fullText.endsWith('\n')) out.write('\n');
    }
    firstGroup = false;
  }
}

bool _isOnlyCommentsAndWhitespace(String s) {
  var i = 0;
  while (i < s.length) {
    final c = s[i];
    if (c == ' ' || c == '\t' || c == '\n' || c == '\r') {
      i++;
      continue;
    }
    if (i + 1 < s.length && s[i] == '/' && s[i + 1] == '/') {
      while (i < s.length && s[i] != '\n') {
        i++;
      }
      continue;
    }
    if (i + 1 < s.length && s[i] == '/' && s[i + 1] == '*') {
      final end = s.indexOf('*/', i + 2);
      if (end == -1) return false;
      i = end + 2;
      continue;
    }
    return false;
  }
  return true;
}
