import 'dart:math' as math;

import 'package:flutter/foundation.dart';

import 'grid_position.dart';
import 'occupancy_grid.dart';

/// A tile reduced to what the packing algorithm needs: an identity and a span.
@immutable
class LayoutTile {
  /// Creates a tile spanning [width] columns and [height] rows.
  const LayoutTile({
    required this.key,
    required this.width,
    required this.height,
  });

  /// Identity of the tile, unique within a single grid.
  final Key key;

  /// Span along the cross axis, in columns.
  final int width;

  /// Span along the main axis, in rows.
  final int height;

  @override
  String toString() => 'LayoutTile($key, ${width}x$height)';
}

/// The result of a packing pass: where every tile sits and how tall the grid
/// ended up being.
@immutable
class GridLayout {
  /// Creates a layout from resolved [placements] spanning [rows] rows.
  const GridLayout({required this.placements, required this.rows});

  /// An empty layout with no tiles and no height.
  static const GridLayout empty = GridLayout(
    placements: <Key, GridPosition>{},
    rows: 0,
  );

  /// Top-left cell of every tile, keyed by [LayoutTile.key].
  final Map<Key, GridPosition> placements;

  /// Total number of rows spanned by the layout.
  final int rows;

  /// The placement of [key], or `null` when the tile is not part of the layout.
  GridPosition? operator [](Key key) => placements[key];

  @override
  String toString() => 'GridLayout(rows: $rows, placements: $placements)';
}

/// Packs [tiles] into [columns] using a first-fit dense strategy.
///
/// Tiles are placed in list order, each one taking the topmost/leftmost slot it
/// fits into, which is what makes the visual order match the list order while
/// still filling holes left by taller neighbours.
///
/// [pinned] forces specific tiles to a fixed cell — used during a drag to hold
/// the dragged tile under the pointer while everything else flows around it. A
/// pin that cannot be honoured (because an earlier pin already took the space)
/// is ignored and that tile falls back to normal placement.
///
/// Spans are clamped to the grid, so a tile wider than [columns] is narrowed
/// rather than dropped. The function never fails: every tile always gets a
/// placement.
GridLayout packDense({
  required List<LayoutTile> tiles,
  required int columns,
  Map<Key, GridPosition> pinned = const <Key, GridPosition>{},
}) {
  if (tiles.isEmpty) return GridLayout.empty;

  final int gridColumns = columns.clamp(1, OccupancyGrid.maxColumns);
  final OccupancyGrid grid = OccupancyGrid(gridColumns);
  final Map<Key, GridPosition> placements = <Key, GridPosition>{};

  int totalArea = 0;
  for (final LayoutTile tile in tiles) {
    totalArea += _widthOf(tile, gridColumns) * _heightOf(tile);
  }
  // Upper bound on the rows a first-fit pass can need: a perfectly packed grid
  // plus one extra row per tile to absorb fragmentation.
  final int rowLimit = (totalArea / gridColumns).ceil() + tiles.length;

  // Pinned tiles claim their slot first so the rest can flow around them.
  if (pinned.isNotEmpty) {
    for (final LayoutTile tile in tiles) {
      final GridPosition? anchor = pinned[tile.key];
      if (anchor == null) continue;
      final int width = _widthOf(tile, gridColumns);
      final int height = _heightOf(tile);
      final int row = math.max(0, anchor.row);
      final int col = anchor.col.clamp(0, gridColumns - width);
      if (!grid.fits(row, col, width, height)) continue;
      grid.place(row, col, width, height);
      placements[tile.key] = (row: row, col: col);
    }
  }

  for (final LayoutTile tile in tiles) {
    if (placements.containsKey(tile.key)) continue;
    final int width = _widthOf(tile, gridColumns);
    final int height = _heightOf(tile);
    final GridPosition slot =
        grid.findFreeSlot(width: width, height: height, rowLimit: rowLimit) ??
        (row: grid.rows, col: 0);
    grid.place(slot.row, slot.col, width, height);
    placements[tile.key] = slot;
  }

  return GridLayout(placements: placements, rows: grid.rows);
}

/// Orders placements the way a reader scans the grid: top to bottom, then left
/// to right. This is the order the widget reports through `onReorder`.
int compareRowMajor(GridPosition a, GridPosition b) {
  final int rowComparison = a.row.compareTo(b.row);
  return rowComparison != 0 ? rowComparison : a.col.compareTo(b.col);
}

int _widthOf(LayoutTile tile, int columns) => tile.width.clamp(1, columns);

int _heightOf(LayoutTile tile) => math.max(1, tile.height);
