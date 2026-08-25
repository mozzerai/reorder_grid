import 'dart:io';

import 'package:flutter_test/flutter_test.dart';
import 'package:path/path.dart' as p;

const String _ownPackagePrefix = 'package:reorder_grid/';

final RegExp _directiveRe = RegExp(
  r'''(^|\n)([ \t]*)(import|export|part)\s+([^;]+);''',
);

final RegExp _uriRe = RegExp(r'''(['"])([^'"]+)\1''');

int _kindOrder(String keyword) {
  switch (keyword) {
    case 'import':
      return 0;
    case 'export':
      return 1;
    case 'part':
      return 2;
  }
  return 3;
}

int _bucket(String uri) {
  if (uri.startsWith('dart:')) return 0;
  if (uri.startsWith(_ownPackagePrefix)) return 2;
  if (uri.startsWith('package:')) return 1;
  return 3;
}

void main() {
  test(
    'directives in lib/ are ordered: imports → exports → parts; '
    'dart → package → package:reorder_grid; alphabetical within group',
    () async {
      final libDir = Directory(p.join(Directory.current.path, 'lib'));
      expect(libDir.existsSync(), isTrue, reason: 'lib/ must exist');

      final violations = <String>[];

      final files = libDir
          .listSync(recursive: true)
          .whereType<File>()
          .where((f) => f.path.endsWith('.dart'));

      for (final file in files) {
        final content = await file.readAsString();
        final matches = _directiveRe.allMatches(content).toList();
        if (matches.isEmpty) continue;

        final relPath = p.relative(file.path, from: Directory.current.path);
        int? prevKind;
        int? prevBucket;
        String? prevUri;

        for (final m in matches) {
          final keyword = m.group(3)!;
          final body = m.group(4)!;
          final uri = _uriRe.firstMatch(body)?.group(2) ?? '';
          final kind = _kindOrder(keyword);
          final bucket = _bucket(uri);

          if (prevKind != null && kind < prevKind) {
            violations.add(
              '$relPath: `$keyword \'$uri\'` appears after a $prevKind-kind '
              'directive (imports → exports → parts required)',
            );
          } else if (prevKind == kind) {
            if (prevBucket != null && bucket < prevBucket) {
              violations.add(
                '$relPath: `$keyword \'$uri\'` (bucket $bucket) appears after '
                'bucket $prevBucket (dart → package → package:reorder_grid required)',
              );
            } else if (prevBucket == bucket &&
                prevUri != null &&
                uri.compareTo(prevUri) < 0) {
              violations.add(
                '$relPath: `$keyword \'$uri\'` is not alphabetical after '
                '`$prevUri`',
              );
            }
          }

          prevKind = kind;
          prevBucket = bucket;
          prevUri = uri;
        }
      }

      expect(
        violations,
        isEmpty,
        reason:
            'Directives under lib/ must follow the project ordering. '
            'Run `dart run tool/organize_imports.dart` to fix.\n'
            'Violations:\n${violations.map((v) => '  - $v').join('\n')}',
      );
    },
  );
}
