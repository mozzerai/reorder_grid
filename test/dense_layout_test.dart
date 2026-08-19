import 'package:flutter/foundation.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:reorder_grid/src/dense_layout.dart';
import 'package:reorder_grid/src/grid_position.dart';

LayoutTile tile(String id, {int width = 1, int height = 1}) =>
    LayoutTile(key: ValueKey<String>(id), width: width, height: height);

Key k(String id) => ValueKey<String>(id);

void main() {
  group('packDense', () {
    test('should return an empty layout for no tiles', () {
      final GridLayout layout = packDense(tiles: <LayoutTile>[], columns: 4);

      expect(layout.placements, isEmpty);
      expect(layout.rows, 0);
    });

    test('should lay tiles out in list order, left to right', () {
      final GridLayout layout = packDense(
        tiles: <LayoutTile>[tile('a'), tile('b'), tile('c')],
        columns: 2,
      );

      expect(layout[k('a')], (row: 0, col: 0));
      expect(layout[k('b')], (row: 0, col: 1));
      expect(layout[k('c')], (row: 1, col: 0));
      expect(layout.rows, 2);
    });

    test('should backfill the hole left by a taller tile', () {
      final GridLayout layout = packDense(
        tiles: <LayoutTile>[
          tile('tall', height: 2),
          tile('wide', width: 3),
          tile('small'),
        ],
        columns: 4,
      );

      expect(layout[k('tall')], (row: 0, col: 0));
      expect(layout[k('wide')], (row: 0, col: 1));
      expect(layout[k('small')], (row: 1, col: 1));
      expect(layout.rows, 2);
    });

    test('should narrow a tile wider than the grid instead of dropping it', () {
      final GridLayout layout = packDense(
        tiles: <LayoutTile>[tile('huge', width: 8), tile('next')],
        columns: 2,
      );

      expect(layout[k('huge')], (row: 0, col: 0));
      expect(layout[k('next')], (row: 1, col: 0));
    });

    test('should treat non-positive spans as a single cell', () {
      final GridLayout layout = packDense(
        tiles: <LayoutTile>[tile('zero', width: 0, height: 0), tile('next')],
        columns: 2,
      );

      expect(layout[k('zero')], (row: 0, col: 0));
      expect(layout[k('next')], (row: 0, col: 1));
      expect(layout.rows, 1);
    });

    test('should place every tile even in a single-column grid', () {
      final GridLayout layout = packDense(
        tiles: <LayoutTile>[tile('a'), tile('b'), tile('c')],
        columns: 1,
      );

      expect(layout.placements, hasLength(3));
      expect(layout.rows, 3);
    });
  });

  group('packDense with pinned tiles', () {
    test('should hold the pinned tile and flow the rest around it', () {
      final GridLayout layout = packDense(
        tiles: <LayoutTile>[tile('a'), tile('b'), tile('c')],
        columns: 2,
        pinned: <Key, GridPosition>{k('c'): (row: 0, col: 0)},
      );

      expect(layout[k('c')], (row: 0, col: 0));
      expect(layout[k('a')], (row: 0, col: 1));
      expect(layout[k('b')], (row: 1, col: 0));
    });

    test('should clamp a pin that overflows the right edge', () {
      final GridLayout layout = packDense(
        tiles: <LayoutTile>[tile('wide', width: 2), tile('a')],
        columns: 3,
        pinned: <Key, GridPosition>{k('wide'): (row: 0, col: 2)},
      );

      expect(layout[k('wide')], (row: 0, col: 1));
    });

    test('should clamp a pin above the first row', () {
      final GridLayout layout = packDense(
        tiles: <LayoutTile>[tile('a')],
        columns: 2,
        pinned: <Key, GridPosition>{k('a'): (row: -3, col: 0)},
      );

      expect(layout[k('a')], (row: 0, col: 0));
    });

    test('should ignore a pin already taken by an earlier pin', () {
      final GridLayout layout = packDense(
        tiles: <LayoutTile>[tile('a'), tile('b')],
        columns: 2,
        pinned: <Key, GridPosition>{
          k('a'): (row: 0, col: 0),
          k('b'): (row: 0, col: 0),
        },
      );

      expect(layout[k('a')], (row: 0, col: 0));
      expect(layout[k('b')], (row: 0, col: 1));
    });

    test('should open a new bottom row when pinned past the content', () {
      final GridLayout layout = packDense(
        tiles: <LayoutTile>[tile('a'), tile('b')],
        columns: 2,
        pinned: <Key, GridPosition>{k('a'): (row: 3, col: 1)},
      );

      expect(layout[k('a')], (row: 3, col: 1));
      expect(layout[k('b')], (row: 0, col: 0));
      expect(layout.rows, 4);
    });
  });

  group('packDense stability', () {
    test('should be idempotent when re-run over its own reading order', () {
      final List<LayoutTile> tiles = <LayoutTile>[
        tile('tall', height: 2),
        tile('wide', width: 3),
        tile('small'),
        tile('other'),
      ];

      final GridLayout first = packDense(tiles: tiles, columns: 4);
      final List<LayoutTile> readingOrder = tiles.toList()
        ..sort(
          (LayoutTile a, LayoutTile b) =>
              compareRowMajor(first[a.key]!, first[b.key]!),
        );
      final GridLayout second = packDense(tiles: readingOrder, columns: 4);

      expect(second.placements, first.placements);
      expect(second.rows, first.rows);
    });
  });

  group('compareRowMajor', () {
    test('should order by row before column', () {
      expect(compareRowMajor((row: 0, col: 5), (row: 1, col: 0)), isNegative);
      expect(compareRowMajor((row: 1, col: 0), (row: 1, col: 2)), isNegative);
      expect(compareRowMajor((row: 2, col: 1), (row: 2, col: 1)), isZero);
    });
  });
}
