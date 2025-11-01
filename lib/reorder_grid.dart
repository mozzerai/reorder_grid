import 'dart:math';
import 'package:flutter/foundation.dart';
import 'package:flutter/material.dart';

/// Represents a tile within the [ReorderGrid].
///
/// Each tile has a specific size in terms of grid cells.
class ReorderGridTile {
  final Key key;
  final int mainAxisCellCount;
  final int crossAxisCellCount;
  final Widget child;

  const ReorderGridTile.count({
    required this.key,
    this.mainAxisCellCount = 1,
    this.crossAxisCellCount = 1,
    required this.child,
  });

  @override
  bool operator ==(Object other) =>
      identical(this, other) ||
      other is ReorderGridTile &&
          runtimeType == other.runtimeType &&
          key == other.key &&
          mainAxisCellCount == other.mainAxisCellCount &&
          crossAxisCellCount == other.crossAxisCellCount;

  @override
  int get hashCode =>
      key.hashCode ^ mainAxisCellCount.hashCode ^ crossAxisCellCount.hashCode;
}

/// A grid that allows users to reorder its children through dragging.
class ReorderGrid extends StatefulWidget {
  final int crossAxisCount;
  final double mainAxisSpacing;
  final double crossAxisSpacing;
  final bool enableReorder;
  final bool showSlotBorders;
  final void Function(int oldIndex, int newIndex)? onReorder;
  final List<ReorderGridTile> children;
  final double borderRadius;

  const ReorderGrid.count({
    super.key,
    required this.crossAxisCount,
    this.mainAxisSpacing = 8.0,
    this.crossAxisSpacing = 8.0,
    this.enableReorder = true,
    this.showSlotBorders = true,
    this.onReorder,
    required this.children,
    this.borderRadius = 12.0,
  });

  @override
  State<ReorderGrid> createState() => _ReorderGridState();
}

/// Internal, mutable representation of a tile for layout calculations.
class _InternalTile {
  final Key key;
  final Widget child;
  final int width;
  final int height;
  int row;
  int col;

  _InternalTile({
    required this.key,
    required this.child,
    required this.width,
    required this.height,
    required this.row,
    required this.col,
  });

  factory _InternalTile.fromReorderGridTile(ReorderGridTile tile) {
    return _InternalTile(
      key: tile.key,
      child: tile.child,
      width: tile.crossAxisCellCount,
      height: tile.mainAxisCellCount,
      row: 0, // Initial position, calculated later.
      col: 0, // Initial position, calculated later.
    );
  }
}

/// Helper class to manage grid cell occupancy for dense packing.
class _OccGrid {
  final int cols;
  int rows = 0;
  final Set<Point<int>> _used = {};

  _OccGrid(this.cols);

  bool _isUsed(int r, int c) => _used.contains(Point(r, c));

  bool fits(int r, int c, int w, int h) {
    if (r < 0 || c < 0 || c + w > cols) return false;
    for (int rr = r; rr < r + h; rr++) {
      for (int cc = c; cc < c + w; cc++) {
        if (_isUsed(rr, cc)) return false;
      }
    }
    return true;
  }

  void place(int r, int c, int w, int h) {
    rows = max(rows, r + h);
    for (int rr = r; rr < r + h; rr++) {
      for (int cc = c; cc < c + w; cc++) {
        _used.add(Point(rr, cc));
      }
    }
  }

  Iterable<Point<int>> scan({required int rowsLimit}) sync* {
    for (int r = 0; r <= rowsLimit; r++) {
      for (int c = 0; c < cols; c++) {
        yield Point(r, c);
      }
    }
  }
}

class _ReorderGridState extends State<ReorderGrid> {
  late List<_InternalTile> _internalTiles;
  Key? _draggingKey;

  @override
  void initState() {
    super.initState();
    _internalTiles =
        widget.children.map(_InternalTile.fromReorderGridTile).toList();
    WidgetsBinding.instance.addPostFrameCallback((_) => _reflow());
  }

  @override
  void didUpdateWidget(ReorderGrid oldWidget) {
    super.didUpdateWidget(oldWidget);
    if (widget.crossAxisCount != oldWidget.crossAxisCount ||
        !listEquals(widget.children, oldWidget.children)) {
      _internalTiles =
          widget.children.map(_InternalTile.fromReorderGridTile).toList();
      _reflow();
    }
  }

  void _reflow() {
    final layout = _layoutDense(tiles: _internalTiles, fixed: const {});
    if (layout != null) {
      _applyLayout(layout);
    }
  }

  void _applyLayout(Map<Key, Point<int>> placements) {
    if (!mounted) return;
    setState(() {
      for (final tile in _internalTiles) {
        final pos = placements[tile.key];
        if (pos != null) {
          tile.row = pos.x;
          tile.col = pos.y;
        }
      }
    });
  }

  void _handleReorder(Key draggedKey, Point<int> targetPosition) {
    final layout = _layoutDense(
      tiles: _internalTiles,
      fixed: {draggedKey: targetPosition},
    );
    if (layout == null) return;

    final oldIndex = widget.children.indexWhere((c) => c.key == draggedKey);
    if (oldIndex == -1) return;

    final positionedTiles = layout.entries.toList()
      ..sort((a, b) {
        final rowCmp = a.value.x.compareTo(b.value.x);
        return rowCmp != 0 ? rowCmp : a.value.y.compareTo(b.value.y);
      });

    final newIndex =
        positionedTiles.indexWhere((entry) => entry.key == draggedKey);

    if (newIndex != -1 && oldIndex != newIndex) {
      widget.onReorder?.call(oldIndex, newIndex);
    }
  }

  _InternalTile? _getTileByKey(Key key) {
    try {
      return _internalTiles.firstWhere((t) => t.key == key);
    } catch (_) {
      return null;
    }
  }

  int get _minCols =>
      _internalTiles.fold<int>(1, (acc, t) => max(acc, t.width));

  int get _currentRows {
    if (_internalTiles.isEmpty) return 0;
    return _internalTiles.map((t) => t.row + t.height).fold<int>(0, max);
  }

  Map<Key, Point<int>>? _layoutDense({
    required List<_InternalTile> tiles,
    required Map<Key, Point<int>> fixed,
  }) {
    final internalCols = widget.crossAxisCount;
    if (internalCols < _minCols) return null;

    final grid = _OccGrid(internalCols);
    final placements = <Key, Point<int>>{};
    final totalTileHeight = tiles.fold<int>(0, (sum, t) => sum + t.height);
    final rowsLimit = (totalTileHeight * 2 / internalCols).ceil() + 10;

    for (final entry in fixed.entries) {
      final tile = _getTileByKey(entry.key);
      if (tile == null) continue;
      final pos = entry.value;
      if (!grid.fits(pos.x, pos.y, tile.width, tile.height)) return null;
      grid.place(pos.x, pos.y, tile.width, tile.height);
      placements[tile.key] = pos;
    }

    final others = [...tiles]
      ..removeWhere((t) => placements.containsKey(t.key))
      ..sort((a, b) {
        final rowCmp = a.row.compareTo(b.row);
        return rowCmp != 0 ? rowCmp : a.col.compareTo(b.col);
      });

    for (final tile in others) {
      bool placed = false;
      for (final pos in grid.scan(rowsLimit: rowsLimit)) {
        if (grid.fits(pos.x, pos.y, tile.width, tile.height)) {
          grid.place(pos.x, pos.y, tile.width, tile.height);
          placements[tile.key] = Point(pos.x, pos.y);
          placed = true;
          break;
        }
      }
      if (!placed) return null;
    }
    return placements;
  }

  @override
  Widget build(BuildContext context) {
    return LayoutBuilder(
      builder: (context, constraints) {
        if (!constraints.hasBoundedWidth) {
          return const Center(
            child: Text('ReorderGrid requires bounded width.'),
          );
        }

        final totalHorizontalSpacing =
            (widget.crossAxisCount - 1) * widget.crossAxisSpacing;
        final cellWidth = (constraints.maxWidth - totalHorizontalSpacing) /
            widget.crossAxisCount;
        final cellHeight = cellWidth;

        final rows = _currentRows;
        final totalVerticalSpacing =
            (rows > 0 ? rows - 1 : 0) * widget.mainAxisSpacing;
        final gridHeight = rows * cellHeight + totalVerticalSpacing;

        return SizedBox(
          width: constraints.maxWidth,
          height: gridHeight,
          child: Stack(
            children: [
              if (widget.enableReorder && rows > 0)
                ..._buildCellDropTargets(
                    rows, widget.crossAxisCount, cellWidth, cellHeight),
              ..._internalTiles.map(
                (tile) => _buildPositionedTile(tile, cellWidth, cellHeight),
              ),
            ],
          ),
        );
      },
    );
  }

  List<Widget> _buildCellDropTargets(
      int rows, int cols, double cellWidth, double cellHeight) {
    return List.generate(rows * cols, (index) {
      final r = index ~/ cols;
      final c = index % cols;
      final left = c * (cellWidth + widget.crossAxisSpacing);
      final top = r * (cellHeight + widget.mainAxisSpacing);

      return Positioned(
        left: left,
        top: top,
        width: cellWidth,
        height: cellHeight,
        child: DragTarget<Key>(
          onWillAcceptWithDetails: (_) => _draggingKey != null,
          onAcceptWithDetails: (details) =>
              _handleReorder(details.data, Point(r, c)),
          builder: (context, candidate, rejected) {
            final hovering = candidate.isNotEmpty;
            return AnimatedContainer(
              duration: const Duration(milliseconds: 150),
              decoration: BoxDecoration(
                color: hovering
                    ? Theme.of(context).primaryColor.withOpacity(0.1)
                    : Colors.transparent,
                border: hovering
                    ? Border.all(
                        color: Theme.of(context).primaryColor.withOpacity(0.5),
                      )
                    : widget.showSlotBorders
                        ? Border.all(color: Colors.grey.withOpacity(0.15))
                        : null,
                borderRadius: BorderRadius.circular(widget.borderRadius),
              ),
            );
          },
        ),
      );
    });
  }

  Widget _buildPositionedTile(
      _InternalTile tile, double cellWidth, double cellHeight) {
    final left = tile.col * (cellWidth + widget.crossAxisSpacing);
    final top = tile.row * (cellHeight + widget.mainAxisSpacing);
    final w = tile.width * cellWidth +
        (tile.width - 1).clamp(0, 100) * widget.crossAxisSpacing;
    final h = tile.height * cellHeight +
        (tile.height - 1).clamp(0, 100) * widget.mainAxisSpacing;

    final tileContent = ClipRRect(
      borderRadius: BorderRadius.circular(widget.borderRadius),
      child: tile.child,
    );

    if (!widget.enableReorder) {
      return Positioned(
        left: left,
        top: top,
        width: w,
        height: h,
        child: tileContent,
      );
    }

    final isDragging = _draggingKey == tile.key;
    final draggableWidget = LongPressDraggable<Key>(
      data: tile.key,
      onDragStarted: () => setState(() => _draggingKey = tile.key),
      onDragEnd: (_) => setState(() => _draggingKey = null),
      feedback: SizedBox(
        width: w,
        height: h,
        child: Material(
          elevation: 4.0,
          color: Colors.transparent,
          borderRadius: BorderRadius.circular(widget.borderRadius),
          child: tileContent,
        ),
      ),
      childWhenDragging: const SizedBox.shrink(),
      child: tileContent,
    );

    return Positioned(
      left: left,
      top: top,
      width: w,
      height: h,
      child: DragTarget<Key>(
        onWillAcceptWithDetails: (details) => details.data != tile.key,
        onAcceptWithDetails: (details) =>
            _handleReorder(details.data, Point(tile.row, tile.col)),
        builder: (context, candidate, rejected) {
          return Stack(
            fit: StackFit.expand,
            children: [
              AnimatedOpacity(
                duration: const Duration(milliseconds: 200),
                opacity: isDragging ? 0.5 : 1.0,
                child: draggableWidget,
              ),
              IgnorePointer(
                child: AnimatedContainer(
                  duration: const Duration(milliseconds: 200),
                  decoration: BoxDecoration(
                    borderRadius: BorderRadius.circular(widget.borderRadius),
                    border: candidate.isNotEmpty
                        ? Border.all(
                            color: Theme.of(context).primaryColor, width: 2)
                        : null,
                  ),
                ),
              ),
            ],
          );
        },
      ),
    );
  }
}
