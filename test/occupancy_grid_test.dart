import 'package:flutter_test/flutter_test.dart';
import 'package:reorder_grid/src/occupancy_grid.dart';

void main() {
  group('OccupancyGrid.fits', () {
    test('should accept a block inside an empty grid', () {
      final OccupancyGrid grid = OccupancyGrid(4);

      expect(grid.fits(0, 0, 4, 2), isTrue);
      expect(grid.fits(10, 3, 1, 1), isTrue);
    });

    test('should reject a block crossing the right edge', () {
      final OccupancyGrid grid = OccupancyGrid(4);

      expect(grid.fits(0, 2, 3, 1), isFalse);
    });

    test('should reject negative coordinates and empty spans', () {
      final OccupancyGrid grid = OccupancyGrid(4);

      expect(grid.fits(-1, 0, 1, 1), isFalse);
      expect(grid.fits(0, -1, 1, 1), isFalse);
      expect(grid.fits(0, 0, 0, 1), isFalse);
      expect(grid.fits(0, 0, 1, 0), isFalse);
    });

    test('should reject a block overlapping an occupied cell', () {
      final OccupancyGrid grid = OccupancyGrid(4)..place(0, 1, 2, 2);

      expect(grid.fits(0, 0, 2, 1), isFalse);
      expect(grid.fits(1, 2, 2, 1), isFalse);
      expect(grid.fits(0, 3, 1, 2), isTrue);
      expect(grid.fits(2, 0, 4, 1), isTrue);
    });

    test('should not mutate the grid', () {
      final OccupancyGrid grid = OccupancyGrid(4);

      grid.fits(20, 0, 1, 1);

      expect(grid.rows, 0);
    });
  });

  group('OccupancyGrid.place', () {
    test('should grow the row count to cover the block', () {
      final OccupancyGrid grid = OccupancyGrid(4)..place(2, 0, 1, 3);

      expect(grid.rows, 5);
    });

    test('should keep the tallest extent when placing higher up', () {
      final OccupancyGrid grid = OccupancyGrid(4)
        ..place(0, 0, 1, 4)
        ..place(0, 1, 1, 1);

      expect(grid.rows, 4);
    });

    test('should pull an out-of-bounds block back inside the grid', () {
      final OccupancyGrid grid = OccupancyGrid(4)..place(0, 3, 2, 1);

      expect(grid.rowMask(0), (1 << 2) | (1 << 3));
    });

    test('should advance firstOpenRow past saturated rows', () {
      final OccupancyGrid grid = OccupancyGrid(2)
        ..place(0, 0, 2, 1)
        ..place(1, 0, 1, 1);

      expect(grid.firstOpenRow, 1);
    });
  });

  group('OccupancyGrid.findFreeSlot', () {
    test('should return the top-left slot of an empty grid', () {
      final OccupancyGrid grid = OccupancyGrid(4);

      expect(grid.findFreeSlot(width: 2, height: 1, rowLimit: 4), (
        row: 0,
        col: 0,
      ));
    });

    test('should fill a hole left by a taller neighbour', () {
      final OccupancyGrid grid = OccupancyGrid(4)
        ..place(0, 0, 2, 2)
        ..place(0, 2, 2, 1);

      expect(grid.findFreeSlot(width: 2, height: 1, rowLimit: 4), (
        row: 1,
        col: 2,
      ));
    });

    test('should skip a row that cannot host the requested width', () {
      final OccupancyGrid grid = OccupancyGrid(4)..place(0, 0, 3, 1);

      expect(grid.findFreeSlot(width: 2, height: 1, rowLimit: 4), (
        row: 1,
        col: 0,
      ));
    });

    test('should return null when the block is wider than the grid', () {
      final OccupancyGrid grid = OccupancyGrid(4);

      expect(grid.findFreeSlot(width: 5, height: 1, rowLimit: 4), isNull);
    });

    test('should return null when nothing fits below the row limit', () {
      final OccupancyGrid grid = OccupancyGrid(1)
        ..place(0, 0, 1, 1)
        ..place(1, 0, 1, 1);

      expect(grid.findFreeSlot(width: 1, height: 1, rowLimit: 1), isNull);
    });
  });
}
