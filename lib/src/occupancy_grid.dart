import 'grid_position.dart';

/// Bitmask-based occupancy map used by the dense packing algorithm.
///
/// Each row is stored as a single integer where bit `c` is set when column `c`
/// is occupied. That keeps the whole structure allocation free (one `int` per
/// row) and turns overlap checks into a couple of bitwise operations.
///
/// Rows are created lazily: any row beyond [rows] is implicitly empty, so the
/// grid can grow downwards without bounds.
class OccupancyGrid {
  /// Creates an empty grid with [columns] columns.
  ///
  /// [columns] must be between 1 and [maxColumns]; values outside that range
  /// are clamped so a misconfigured widget degrades instead of crashing in
  /// release mode, where asserts are stripped.
  OccupancyGrid(int columns)
    : assert(
        columns >= 1 && columns <= maxColumns,
        'columns must be within 1..$maxColumns, got $columns',
      ),
      columns = columns.clamp(1, maxColumns),
      _fullRowMask = (1 << columns.clamp(1, maxColumns)) - 1;

  /// The widest grid this implementation supports.
  ///
  /// One row is a single Dart `int`; 62 columns stay inside the 63-bit signed
  /// range on every platform, including the web (where `int` is a JS number
  /// only up to 2^53, hence the conservative bound).
  static const int maxColumns = 62;

  /// Number of columns in the grid.
  final int columns;

  /// Mask with every column bit set, used to skip saturated rows.
  final int _fullRowMask;

  final List<int> _rowMasks = <int>[];

  int _rows = 0;
  int _firstOpenRow = 0;

  /// Number of rows currently spanned by placed tiles.
  int get rows => _rows;

  /// The lowest row index that still has at least one free cell.
  int get firstOpenRow => _firstOpenRow;

  int _maskAt(int row) => row < _rowMasks.length ? _rowMasks[row] : 0;

  int _spanMask(int col, int width) => ((1 << width) - 1) << col;

  /// Whether a `width` x `height` block fits with its top-left corner at
  /// (`row`, `col`) without overlapping an occupied cell.
  ///
  /// This is a pure read: rows past the end of the grid count as free.
  bool fits(int row, int col, int width, int height) {
    if (row < 0 || col < 0 || width <= 0 || height <= 0) return false;
    if (col + width > columns) return false;
    final int mask = _spanMask(col, width);
    for (int r = row; r < row + height; r++) {
      if (_maskAt(r) & mask != 0) return false;
    }
    return true;
  }

  /// Marks a `width` x `height` block as occupied at (`row`, `col`).
  ///
  /// Callers are expected to have checked [fits] first; placing over an
  /// occupied cell is a no-op for that cell rather than an error.
  void place(int row, int col, int width, int height) {
    if (row < 0 || col < 0 || width <= 0 || height <= 0) return;
    final int clampedWidth = width > columns ? columns : width;
    final int clampedCol = col + clampedWidth > columns
        ? columns - clampedWidth
        : col;
    final int bottom = row + height;
    while (_rowMasks.length < bottom) {
      _rowMasks.add(0);
    }
    final int mask = _spanMask(clampedCol, clampedWidth);
    for (int r = row; r < bottom; r++) {
      _rowMasks[r] |= mask;
    }
    if (bottom > _rows) _rows = bottom;
    while (_firstOpenRow < _rowMasks.length &&
        _rowMasks[_firstOpenRow] == _fullRowMask) {
      _firstOpenRow++;
    }
  }

  /// Finds the first free slot for a `width` x `height` block, scanning in
  /// row-major order and skipping fully occupied rows.
  ///
  /// Returns `null` when nothing fits at or above `rowLimit`.
  GridPosition? findFreeSlot({
    required int width,
    required int height,
    required int rowLimit,
  }) {
    if (width <= 0 || height <= 0 || width > columns) return null;
    final int baseMask = (1 << width) - 1;
    for (int row = _firstOpenRow; row <= rowLimit; row++) {
      if (_maskAt(row) == _fullRowMask) continue;
      for (int col = 0; col + width <= columns; col++) {
        final int mask = baseMask << col;
        bool free = true;
        for (int r = row; r < row + height; r++) {
          if (_maskAt(r) & mask != 0) {
            free = false;
            break;
          }
        }
        if (free) return (row: row, col: col);
      }
    }
    return null;
  }

  /// Debug helper: the occupancy bitmask of [row].
  int rowMask(int row) => _maskAt(row);
}
