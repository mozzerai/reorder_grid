// Rewrites all relative `import`/`export` URIs in reorder_grid/lib/**.dart to
// `package:reorder_grid/...`. Skips `part`/`part of` (kept relative by convention),
// `dart:` and `package:` URIs. Handles directives wrapped across lines.
//
// Run from the package root:
//   dart run tool/rewrite_imports.dart

import 'dart:io';

import 'package:path/path.dart' as p;

const String _packageName = 'reorder_grid';

// Matches `import` or `export` directives. URI in group 3 (quoted by group 2).
// Anchored on a word boundary so we don't match `part` or identifiers.
final RegExp _directiveRe = RegExp(
  r'''(^|\n)([ \t]*)(import|export)\s+(['"])([^'"]+)\4''',
);

Future<void> main(List<String> args) async {
  final libDir = Directory(p.join(Directory.current.path, 'lib'));
  if (!libDir.existsSync()) {
    stderr.writeln('Error: lib/ not found. Run from the package root.');
    exit(1);
  }

  var filesScanned = 0;
  var filesChanged = 0;
  var linesRewritten = 0;

  final files = libDir
      .listSync(recursive: true)
      .whereType<File>()
      .where((f) => f.path.endsWith('.dart'))
      .toList();

  for (final file in files) {
    filesScanned++;
    final original = await file.readAsString();
    var changedInFile = 0;
    final fileDir = p.dirname(file.path);

    final updated = original.replaceAllMapped(_directiveRe, (m) {
      final lead = m.group(1)!;
      final indent = m.group(2)!;
      final keyword = m.group(3)!;
      final quote = m.group(4)!;
      final uri = m.group(5)!;

      if (uri.startsWith('dart:') || uri.startsWith('package:')) {
        return m.group(0)!;
      }

      final resolved = p.normalize(p.join(fileDir, uri));
      var relToLib = p.relative(resolved, from: libDir.path);

      // Some legacy imports have extra `../` segments and resolve outside
      // lib/, yet the intended target (e.g. core/...) clearly lives under
      // lib/. Recover by stripping leading `../` until the tail exists.
      if (relToLib.startsWith('..')) {
        final parts = p.split(uri).where((s) => s != '..' && s != '.').toList();
        String? recovered;
        for (var start = 0; start < parts.length; start++) {
          final candidate = p.joinAll(parts.sublist(start));
          if (File(p.join(libDir.path, candidate)).existsSync()) {
            recovered = candidate;
            break;
          }
        }
        if (recovered == null) {
          stderr.writeln(
            'Warning: ${p.relative(file.path)} resolves outside lib/: $uri',
          );
          return m.group(0)!;
        }
        relToLib = recovered;
      }

      final newUri = 'package:$_packageName/${relToLib.replaceAll(r'\', '/')}';
      changedInFile++;
      return '$lead$indent$keyword $quote$newUri$quote';
    });

    if (changedInFile > 0) {
      await file.writeAsString(updated);
      filesChanged++;
      linesRewritten += changedInFile;
    }
  }

  stdout.writeln('Scanned: $filesScanned files');
  stdout.writeln('Changed: $filesChanged files');
  stdout.writeln('Rewritten: $linesRewritten directives');
}
