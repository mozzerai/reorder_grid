import 'dart:io';

import 'package:flutter_test/flutter_test.dart';
import 'package:path/path.dart' as p;

final _directiveRe = RegExp(
  r'''(^|\n)[ \t]*(import|export)\s+(['"])([^'"]+)\3''',
);

void main() {
  test('all imports/exports in lib/ use package: or dart: URIs', () async {
    final libDir = Directory(p.join(Directory.current.path, 'lib'));
    expect(libDir.existsSync(), isTrue, reason: 'lib/ must exist');

    final violations = <String>[];

    final files = libDir
        .listSync(recursive: true)
        .whereType<File>()
        .where((f) => f.path.endsWith('.dart'));

    for (final file in files) {
      final content = await file.readAsString();
      for (final m in _directiveRe.allMatches(content)) {
        final uri = m.group(4)!;
        if (uri.startsWith('dart:') || uri.startsWith('package:')) continue;

        final prefix = content.substring(0, m.start);
        final lineNumber = '\n'.allMatches(prefix).length + 1;
        final relPath = p.relative(file.path, from: Directory.current.path);
        violations.add('$relPath:$lineNumber - ${m.group(2)} \'$uri\'');
      }
    }

    expect(
      violations,
      isEmpty,
      reason:
          'Imports/exports under lib/ must use package:reorder_grid/... or dart: URIs, '
          'not relative paths. Run `dart run tool/rewrite_imports.dart` to fix.\n'
          'Violations:\n${violations.map((v) => '  - $v').join('\n')}',
    );
  });
}
