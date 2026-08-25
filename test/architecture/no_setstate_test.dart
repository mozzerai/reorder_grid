import 'dart:io';

import 'package:flutter_test/flutter_test.dart';
import 'package:path/path.dart' as p;

/// Architecture rule: `setState` is banned in `lib/`.
///
/// Local widget state must use `ValueNotifier` + `ValueListenableBuilder`
/// (scoped rebuilds), and screen/business state lives in `ChangeNotifier`
/// ViewModels. `setState` rebuilds the whole `State` subtree, which this
/// project avoids — see the ValueNotifier reactivity refactor.
final RegExp _setStateRe = RegExp(r'\bsetState\s*\(');

void main() {
  test('no setState( calls in lib/ — use ValueNotifier/ViewModels instead', () {
    final libDir = Directory(p.join(Directory.current.path, 'lib'));
    expect(libDir.existsSync(), isTrue, reason: 'lib/ must exist');

    final violations = <String>[];

    final files = libDir
        .listSync(recursive: true)
        .whereType<File>()
        .where((f) => f.path.endsWith('.dart'));

    for (final file in files) {
      final relPath = p.relative(file.path, from: Directory.current.path);
      final lines = file.readAsLinesSync();
      for (var i = 0; i < lines.length; i++) {
        final line = lines[i];
        final match = _setStateRe.firstMatch(line);
        if (match == null) continue;
        // Skip matches that sit inside a line comment.
        final commentIdx = line.indexOf('//');
        if (commentIdx != -1 && commentIdx < match.start) continue;
        violations.add('$relPath:${i + 1}: ${line.trim()}');
      }
    }

    expect(
      violations,
      isEmpty,
      reason:
          'setState is banned in lib/. Replace with a ValueNotifier + '
          'ValueListenableBuilder (local UI state) or a ChangeNotifier '
          'ViewModel (screen state). Offending lines:\n'
          '${violations.join('\n')}',
    );
  });
}
