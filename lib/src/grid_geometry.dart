import 'dart:math' as math;
import 'dart:ui' show Offset;

import 'package:flutter/foundation.dart';

import 'grid_position.dart';

/// Converts between grid cells and pixels for a given set of constraints.
///
/// Geometry is derived from the incoming constraints on every build and never
/// stored on the tiles themselves. That is what makes the grid respond to a
/// resize for free: the logical layout stays put, only the pixel mapping moves.
@immutable
class GridGeometry {
  /// Creates a geometry from already resolved cell metrics.
  const GridGeometry({
    required this.columns,
    required this.cellWidth,
    required this.cellHeight,
    required this.mainAxisSpacing,
    required this.crossAxisSpacing,
  });

  /// Derives cell metrics that fill [availableWidth] with [columns] columns.
  ///
  /// [cellAspectRatio] is the cell's width divided by its height, matching
  /// `GridView.childAspectRatio`.
  factory GridGeometry.fromWidth({
    required double availableWidth,
    required int columns,
    required double mainAxisSpacing,
    required double crossAxisSpacing,
    double cellAspectRatio = 1.0,
  }) {
    final int safeColumns = math.max(1, columns);
    final double totalSpacing = (safeColumns - 1) * crossAxisSpacing;
    final double cellWidth = math.max(
      0.0,
      (availableWidth - totalSpacing) / safeColumns,
    );
    final double ratio = cellAspectRatio > 0 ? cellAspectRatio : 1.0;
    return GridGeometry(
      columns: safeColumns,
      cellWidth: cellWidth,
      cellHeight: cellWidth / ratio,
      mainAxisSpacing: mainAxisSpacing,
      crossAxisSpacing: crossAxisSpacing,
    );
  }

  /// Number of columns the geometry was built for.
  final int columns;

  /// Width of a single cell, in logical pixels.
  final double cellWidth;

  /// Height of a single cell, in logical pixels.
  final double cellHeight;

  /// Gap between rows, in logical pixels.
  final double mainAxisSpacing;

  /// Gap between columns, in logical pixels.
  final double crossAxisSpacing;

  /// Distance from the grid's left edge to the start of [column].
  double leftOf(int column) => column * (cellWidth + crossAxisSpacing);

  /// Distance from the grid's top edge to the start of [row].
  double topOf(int row) => row * (cellHeight + mainAxisSpacing);

  /// Pixel width of a block spanning [columnSpan] columns, gaps included.
  double widthOf(int columnSpan) =>
      columnSpan * cellWidth + math.max(0, columnSpan - 1) * crossAxisSpacing;

  /// Pixel height of a block spanning [rowSpan] rows, gaps included.
  double heightOf(int rowSpan) =>
      rowSpan * cellHeight + math.max(0, rowSpan - 1) * mainAxisSpacing;

  /// Pixel height of the whole grid for [rows] rows.
  double totalHeight(int rows) => rows <= 0 ? 0.0 : heightOf(rows);

  /// Maps a grid-local [offset] to the cell under it, clamped to the grid.
  ///
  /// Returns `null` when the geometry is degenerate (zero-sized cells) or the
  /// grid has no rows.
  GridPosition? cellAt(Offset offset, {required int rows}) {
    if (cellWidth <= 0 || cellHeight <= 0 || rows <= 0) return null;
    final int col = (offset.dx / (cellWidth + crossAxisSpacing)).floor().clamp(
      0,
      columns - 1,
    );
    final int row = (offset.dy / (cellHeight + mainAxisSpacing)).floor().clamp(
      0,
      rows - 1,
    );
    return (row: row, col: col);
  }

  @override
  bool operator ==(Object other) =>
      identical(this, other) ||
      other is GridGeometry &&
          columns == other.columns &&
          cellWidth == other.cellWidth &&
          cellHeight == other.cellHeight &&
          mainAxisSpacing == other.mainAxisSpacing &&
          crossAxisSpacing == other.crossAxisSpacing;

  @override
  int get hashCode => Object.hash(
    columns,
    cellWidth,
    cellHeight,
    mainAxisSpacing,
    crossAxisSpacing,
  );

  @override
  String toString() =>
      'GridGeometry(columns: $columns, cell: ${cellWidth}x$cellHeight)';
}
