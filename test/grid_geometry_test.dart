import 'package:flutter_test/flutter_test.dart';
import 'package:reorder_grid/src/grid_geometry.dart';

void main() {
  GridGeometry build({
    double width = 400,
    int columns = 4,
    double spacing = 10,
    double aspectRatio = 1.0,
  }) {
    return GridGeometry.fromWidth(
      availableWidth: width,
      columns: columns,
      mainAxisSpacing: spacing,
      crossAxisSpacing: spacing,
      cellAspectRatio: aspectRatio,
    );
  }

  group('GridGeometry.fromWidth', () {
    test('should split the width minus the gaps across the columns', () {
      final GridGeometry geometry = build();

      // 400 - 3 gaps of 10 = 370, over 4 columns.
      expect(geometry.cellWidth, 92.5);
      expect(geometry.cellHeight, 92.5);
    });

    test('should derive the cell height from the aspect ratio', () {
      final GridGeometry geometry = build(aspectRatio: 2.0);

      expect(geometry.cellHeight, geometry.cellWidth / 2);
    });

    test('should never produce a negative cell width', () {
      final GridGeometry geometry = build(width: 5, columns: 4, spacing: 20);

      expect(geometry.cellWidth, 0);
    });

    test('should fall back to a single column for non-positive counts', () {
      final GridGeometry geometry = build(columns: 0);

      expect(geometry.columns, 1);
      expect(geometry.cellWidth, 400);
    });
  });

  group('GridGeometry pixel mapping', () {
    test('should offset each cell by its stride', () {
      final GridGeometry geometry = build();

      expect(geometry.leftOf(0), 0);
      expect(geometry.leftOf(2), 2 * (92.5 + 10));
      expect(geometry.topOf(3), 3 * (92.5 + 10));
    });

    test('should include the inner gaps in a multi-cell span', () {
      final GridGeometry geometry = build();

      expect(geometry.widthOf(1), 92.5);
      expect(geometry.widthOf(3), 3 * 92.5 + 2 * 10);
      expect(geometry.heightOf(2), 2 * 92.5 + 10);
    });

    test('should measure the total height without a trailing gap', () {
      final GridGeometry geometry = build();

      expect(geometry.totalHeight(0), 0);
      expect(geometry.totalHeight(1), 92.5);
      expect(geometry.totalHeight(2), 2 * 92.5 + 10);
    });
  });

  group('GridGeometry.cellAt', () {
    test('should resolve the cell under an offset', () {
      final GridGeometry geometry = build();

      expect(geometry.cellAt(const Offset(0, 0), rows: 3), (row: 0, col: 0));
      expect(geometry.cellAt(const Offset(210, 105), rows: 3), (
        row: 1,
        col: 2,
      ));
    });

    test('should clamp offsets outside the grid', () {
      final GridGeometry geometry = build();

      expect(geometry.cellAt(const Offset(-50, -50), rows: 3), (
        row: 0,
        col: 0,
      ));
      expect(geometry.cellAt(const Offset(9999, 9999), rows: 3), (
        row: 2,
        col: 3,
      ));
    });

    test('should return null for a degenerate grid', () {
      final GridGeometry geometry = build(width: 5, columns: 4, spacing: 20);

      expect(geometry.cellAt(const Offset(10, 10), rows: 3), isNull);
      expect(build().cellAt(const Offset(10, 10), rows: 0), isNull);
    });
  });

  group('GridGeometry equality', () {
    test('should compare by value', () {
      expect(build(), build());
      expect(build().hashCode, build().hashCode);
      expect(build(), isNot(build(columns: 2)));
    });
  });
}
