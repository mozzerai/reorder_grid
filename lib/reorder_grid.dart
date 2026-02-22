import 'dart:async';
import 'dart:math';
import 'package:flutter/foundation.dart';
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';

/// Grid position with explicit row/col semantics.
typedef _GridPos = ({int row, int col});

/// Represents a tile within the [ReorderGrid].
///
/// Each tile has a specific size in terms of grid cells.
class ReorderGridTile {
  final Key key;
  final int mainAxisCellCount;
  final int crossAxisCellCount;
  final double borderRadius;
  final Widget child;

  const ReorderGridTile.count({
    required this.key,
    this.mainAxisCellCount = 1,
    this.crossAxisCellCount = 1,
    required this.child,
    this.borderRadius = 8.0,
  });

  @override
  bool operator ==(Object other) =>
      identical(this, other) ||
      other is ReorderGridTile &&
          runtimeType == other.runtimeType &&
          key == other.key &&
          mainAxisCellCount == other.mainAxisCellCount &&
          crossAxisCellCount == other.crossAxisCellCount &&
          child == other.child &&
          borderRadius == other.borderRadius;

  @override
  int get hashCode =>
      key.hashCode ^
      mainAxisCellCount.hashCode ^
      crossAxisCellCount.hashCode ^
      child.hashCode ^
      borderRadius.hashCode;
}

/// A grid that allows users to reorder its children through dragging.
///
/// Supports variable-size tiles with smooth animated transitions,
/// live preview during drag, and haptic feedback.
class ReorderGrid extends StatefulWidget {
  final int crossAxisCount;
  final double mainAxisSpacing;
  final double crossAxisSpacing;
  final bool enableReorder;
  final bool showSlotBorders;
  final void Function(int oldIndex, int newIndex)? onReorder;
  final List<ReorderGridTile> children;
  final double borderRadius;
  final Duration animationDuration;

  const ReorderGrid.count({
    super.key,
    required this.crossAxisCount,
    this.mainAxisSpacing = 8.0,
    this.crossAxisSpacing = 8.0,
    this.enableReorder = true,
    this.showSlotBorders = true,
    this.onReorder,
    required this.children,
    this.borderRadius = 8.0,
    this.animationDuration = const Duration(milliseconds: 300),
  });

  @override
  State<ReorderGrid> createState() => _ReorderGridState();
}

/// Internal, mutable representation of a tile for layout calculations.
class _InternalTile {
  final Key key;
  Widget child;
  final int width;
  final int height;
  double left = 0;
  double top = 0;
  double pixelWidth = 0;
  double pixelHeight = 0;

  _InternalTile({
    required this.key,
    required this.child,
    required this.width,
    required this.height,
  });

  factory _InternalTile.fromReorderGridTile(ReorderGridTile tile) {
    return _InternalTile(
      key: tile.key,
      child: tile.child,
      width: tile.crossAxisCellCount,
      height: tile.mainAxisCellCount,
    );
  }
}

/// Bitmask-based occupancy grid for dense packing.
///
/// Each row is represented as a single integer bitmask where bit `c` is set
/// when column `c` is occupied. This eliminates per-cell object allocations
/// and enables fast row-level skip optimisations in [scan].
class _OccGrid {
  final int cols;
  final int _fullRowMask;
  int rows = 0;
  final List<int> _rowMasks = [];

  _OccGrid(this.cols)
      : assert(cols > 0 && cols <= 62),
        _fullRowMask = (1 << cols) - 1;

  void _ensureRows(int needed) {
    while (_rowMasks.length < needed) {
      _rowMasks.add(0);
    }
  }

  /// Bitmask with bits [c..c+w-1] set.
  int _spanMask(int c, int w) => ((1 << w) - 1) << c;

  bool fits(int r, int c, int w, int h) {
    if (r < 0 || c < 0 || c + w > cols) return false;
    final mask = _spanMask(c, w);
    _ensureRows(r + h);
    for (int rr = r; rr < r + h; rr++) {
      if (_rowMasks[rr] & mask != 0) return false;
    }
    return true;
  }

  void place(int r, int c, int w, int h) {
    _ensureRows(r + h);
    rows = max(rows, r + h);
    final mask = _spanMask(c, w);
    for (int rr = r; rr < r + h; rr++) {
      _rowMasks[rr] |= mask;
    }
  }

  Iterable<_GridPos> scan({required int rowsLimit}) sync* {
    _ensureRows(rowsLimit + 1);
    for (int r = 0; r <= rowsLimit; r++) {
      if (_rowMasks[r] == _fullRowMask) continue;
      for (int c = 0; c < cols; c++) {
        if ((_rowMasks[r] >> c) & 1 == 1) continue;
        yield (row: r, col: c);
      }
    }
  }
}

class _ReorderGridState extends State<ReorderGrid> {
  late List<_InternalTile> _internalTiles;
  late Map<Key, _InternalTile> _tileByKey;
  Key? _draggingKey;
  _GridPos? _hoverPosition;
  _GridPos? _lastPreviewPosition;
  bool _dropHandled = false;

  // Cached layout metrics
  double _cellWidth = 0;
  double _cellHeight = 0;

  // Store the pre-drag layout for reverting
  Map<Key, _GridPos>? _preDragLayout;

  // Store the last hover position so _handleDrop can use it
  // even after _onDragEnd fires (onDragEnd fires BEFORE onAcceptWithDetails)
  _GridPos? _lastHoverPosition;

  // Track which tile was just dropped so it appears instantly (no fade-in)
  Key? _justDroppedKey;

  // Debounce hover to reduce sensitivity during drag
  Timer? _hoverDebounce;

  @override
  void initState() {
    super.initState();
    _rebuildTiles();
    WidgetsBinding.instance.addPostFrameCallback((_) => _reflow());
  }

  @override
  void dispose() {
    _hoverDebounce?.cancel();
    super.dispose();
  }

  @override
  void didUpdateWidget(ReorderGrid oldWidget) {
    super.didUpdateWidget(oldWidget);
    final columnsChanged = widget.crossAxisCount != oldWidget.crossAxisCount;
    final childrenChanged = !listEquals(widget.children, oldWidget.children);
    if (columnsChanged || childrenChanged) {
      _syncTiles();
      // Only reflow if columns changed (need full re-layout) or if
      // the children change was NOT from our own drop (e.g. external add/remove).
      // After a drop, tiles already have correct positions — don't fight them.
      if (columnsChanged) {
        WidgetsBinding.instance.addPostFrameCallback((_) {
          if (mounted) _reflow();
        });
      } else if (childrenChanged && !_dropHandled) {
        // External change (not from our drop) — reflow to place new/removed tiles
        _reflow();
      }
    }
  }

  /// Sync internal tiles with new widget.children, preserving pixel positions
  /// for tiles that already exist (so AnimatedPositioned can animate smoothly).
  void _syncTiles() {
    final newTiles = <_InternalTile>[];
    for (final child in widget.children) {
      final existing = _tileByKey[child.key];
      if (existing != null &&
          existing.width == child.crossAxisCellCount &&
          existing.height == child.mainAxisCellCount) {
        // Preserve position, update child widget
        existing.child = child.child;
        newTiles.add(existing);
      } else {
        newTiles.add(_InternalTile.fromReorderGridTile(child));
      }
    }
    _internalTiles = newTiles;
    _tileByKey = {for (final t in _internalTiles) t.key: t};
  }

  void _rebuildTiles() {
    _internalTiles =
        widget.children.map(_InternalTile.fromReorderGridTile).toList();
    _tileByKey = {for (final t in _internalTiles) t.key: t};
  }

  // ── Layout ──────────────────────────────────────────────────────────────

  void _reflow() {
    final layout = _layoutDense(tiles: _internalTiles, fixed: const {});
    if (layout != null) {
      _applyLayout(layout);
    }
  }

  void _applyLayout(Map<Key, _GridPos> placements) {
    if (!mounted) return;
    bool changed = false;
    for (final tile in _internalTiles) {
      final pos = placements[tile.key];
      if (pos == null) continue;
      final newLeft = _cellLeft(pos.col);
      final newTop = _cellTop(pos.row);
      final newW = _spanWidth(tile.width);
      final newH = _spanHeight(tile.height);
      if (tile.left != newLeft ||
          tile.top != newTop ||
          tile.pixelWidth != newW ||
          tile.pixelHeight != newH) {
        changed = true;
        break;
      }
    }
    if (!changed) return;
    setState(() {
      for (final tile in _internalTiles) {
        final pos = placements[tile.key];
        if (pos != null) {
          tile.left = _cellLeft(pos.col);
          tile.top = _cellTop(pos.row);
          tile.pixelWidth = _spanWidth(tile.width);
          tile.pixelHeight = _spanHeight(tile.height);
        }
      }
    });
  }

  Map<Key, _GridPos> _currentLayout() {
    final layout = _layoutDense(tiles: _internalTiles, fixed: const {});
    return layout ?? {};
  }

  // ── Drag handling ───────────────────────────────────────────────────────

  void _onDragStarted(Key key) {
    _preDragLayout = _currentLayout();
    setState(() => _draggingKey = key);
    HapticFeedback.mediumImpact();
  }

  void _onDragEnd() {
    _hoverDebounce?.cancel();
    // Save hover position before clearing — _handleDrop may need it
    _lastHoverPosition = _hoverPosition;

    // In Flutter, onDragEnd fires BEFORE onAcceptWithDetails.
    // Defer cleanup so _handleDrop gets a chance to run first.
    WidgetsBinding.instance.addPostFrameCallback((_) {
      if (!mounted) return;

      // If _handleDrop already ran (it sets _dropHandled = true), skip revert
      if (_dropHandled) {
        _dropHandled = false;
        return;
      }

      // Drop didn't happen — revert to pre-drag layout
      final wasPreview = _lastHoverPosition != null;
      _hoverPosition = null;
      _lastPreviewPosition = null;
      _lastHoverPosition = null;
      _preDragLayout = null;
      setState(() => _draggingKey = null);

      if (wasPreview) {
        _reflow();
      }
    });
  }

  /// Clamp a grid position so the given tile fits within the grid bounds.
  _GridPos _clampForTile(_GridPos position, _InternalTile tile) {
    final maxCol = max(0, widget.crossAxisCount - tile.width);
    final col = position.col.clamp(0, maxCol);
    final row = max(0, position.row);
    return (row: row, col: col);
  }

  void _onHover(_GridPos position) {
    if (_draggingKey == null) return;

    final draggedTile = _getTileByKey(_draggingKey!);
    if (draggedTile == null) return;

    final clamped = _clampForTile(position, draggedTile);
    if (_lastPreviewPosition == clamped) return;

    // Always track the latest hover for _handleDrop
    _hoverPosition = clamped;

    // Debounce the preview layout to reduce sensitivity
    _hoverDebounce?.cancel();
    _hoverDebounce = Timer(const Duration(milliseconds: 150), () {
      if (!mounted || _draggingKey == null) return;

      _lastPreviewPosition = clamped;

      final layout = _layoutDense(
        tiles: _internalTiles,
        fixed: {_draggingKey!: clamped},
      );

      if (layout != null) {
        _applyLayout(layout);
        HapticFeedback.selectionClick();
      }
    });
  }

  void _handleDrop(Key draggedKey, Offset globalOffset) {
    _hoverDebounce?.cancel();
    final tile = _getTileByKey(draggedKey);
    if (tile == null) return;

    // Use hover position from live preview, then saved hover, then compute
    final rawPos =
        _hoverPosition ?? _lastHoverPosition ?? _cellFromGlobal(globalOffset);
    if (rawPos == null) return;

    final hoverPos = _clampForTile(rawPos, tile);

    final layout = _layoutDense(
      tiles: _internalTiles,
      fixed: {draggedKey: hoverPos},
    );
    if (layout == null) return;

    final oldIndex = widget.children.indexWhere((c) => c.key == draggedKey);
    if (oldIndex == -1) return;

    final positionedTiles = layout.entries.toList()
      ..sort((a, b) {
        final rowCmp = a.value.row.compareTo(b.value.row);
        return rowCmp != 0 ? rowCmp : a.value.col.compareTo(b.value.col);
      });

    final newIndex =
        positionedTiles.indexWhere((entry) => entry.key == draggedKey);

    // Apply layout and reorder internal tiles
    _applyLayout(layout);
    final reordered =
        positionedTiles.map((e) => _tileByKey[e.key]!).toList();
    _internalTiles = reordered;
    _tileByKey = {for (final t in _internalTiles) t.key: t};

    // Mark drop as handled so deferred _onDragEnd cleanup doesn't revert
    _dropHandled = true;
    _justDroppedKey = draggedKey;
    _hoverPosition = null;
    _lastPreviewPosition = null;
    _lastHoverPosition = null;
    _preDragLayout = null;
    setState(() => _draggingKey = null);

    // Clear _justDroppedKey after the frame so future rebuilds use normal opacity
    WidgetsBinding.instance.addPostFrameCallback((_) {
      if (mounted && _justDroppedKey != null) {
        setState(() => _justDroppedKey = null);
      }
    });

    if (newIndex != -1 && oldIndex != newIndex) {
      widget.onReorder?.call(oldIndex, newIndex);
    }
  }

  // ── Grid cell from pointer position ─────────────────────────────────────

  _GridPos? _cellFromGlobal(Offset globalOffset) {
    final box = context.findRenderObject() as RenderBox?;
    if (box == null || !box.hasSize) return null;
    final local = box.globalToLocal(globalOffset);
    return _cellFromOffset(local);
  }

  _GridPos? _cellFromOffset(Offset local) {
    if (_cellWidth <= 0 || _cellHeight <= 0) return null;

    final colStride = _cellWidth + widget.crossAxisSpacing;
    final rowStride = _cellHeight + widget.mainAxisSpacing;

    final rows = _currentRows;
    if (rows <= 0) return null;

    // Clamp to grid bounds instead of returning null at edges
    final col = (local.dx / colStride).floor().clamp(0, widget.crossAxisCount - 1);
    final row = (local.dy / rowStride).floor().clamp(0, rows - 1);

    return (row: row, col: col);
  }

  // ── Layout helpers ──────────────────────────────────────────────────────

  _InternalTile? _getTileByKey(Key key) => _tileByKey[key];

  int get _minCols =>
      _internalTiles.fold<int>(1, (acc, t) => max(acc, t.width));

  int get _currentRows {
    if (_internalTiles.isEmpty) return 0;
    int maxRow = 0;
    for (final tile in _internalTiles) {
      final tileBottom =
          ((tile.top + tile.pixelHeight + widget.mainAxisSpacing) /
                  (_cellHeight + widget.mainAxisSpacing))
              .ceil();
      if (tileBottom > maxRow) maxRow = tileBottom;
    }
    return maxRow > 0 ? maxRow : 1;
  }

  Map<Key, _GridPos>? _layoutDense({
    required List<_InternalTile> tiles,
    required Map<Key, _GridPos> fixed,
  }) {
    final internalCols = widget.crossAxisCount;
    if (internalCols < _minCols) return null;

    final grid = _OccGrid(internalCols);
    final placements = <Key, _GridPos>{};
    final totalArea =
        tiles.fold<int>(0, (sum, t) => sum + t.width * t.height);
    final rowsLimit = (totalArea / internalCols).ceil() + tiles.length;

    for (final entry in fixed.entries) {
      final tile = _getTileByKey(entry.key);
      if (tile == null) continue;
      final pos = entry.value;
      if (!grid.fits(pos.row, pos.col, tile.width, tile.height)) return null;
      grid.place(pos.row, pos.col, tile.width, tile.height);
      placements[tile.key] = pos;
    }

    // Preserve list order — tiles are already in the correct sequence
    // (either from widget.children or from a previous reorder)
    final others = tiles
        .where((t) => !placements.containsKey(t.key))
        .toList();

    for (final tile in others) {
      bool placed = false;
      for (final pos in grid.scan(rowsLimit: rowsLimit)) {
        if (grid.fits(pos.row, pos.col, tile.width, tile.height)) {
          grid.place(pos.row, pos.col, tile.width, tile.height);
          placements[tile.key] = pos;
          placed = true;
          break;
        }
      }
      if (!placed) return null;
    }
    return placements;
  }

  // ── Position helpers ──────────────────────────────────────────────────

  double _cellLeft(int col) => col * (_cellWidth + widget.crossAxisSpacing);

  double _cellTop(int row) => row * (_cellHeight + widget.mainAxisSpacing);

  double _spanWidth(int spanCols) =>
      spanCols * _cellWidth + max(0, spanCols - 1) * widget.crossAxisSpacing;

  double _spanHeight(int spanRows) =>
      spanRows * _cellHeight + max(0, spanRows - 1) * widget.mainAxisSpacing;

  // ── Build ─────────────────────────────────────────────────────────────

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
        _cellWidth = (constraints.maxWidth - totalHorizontalSpacing) /
            widget.crossAxisCount;
        _cellHeight = _cellWidth;

        // Layout pass if any tiles have no pixel positions yet (new or reset)
        final needsLayout = _internalTiles.any((t) => t.pixelWidth == 0);
        if (needsLayout) {
          final layout =
              _layoutDense(tiles: _internalTiles, fixed: const {});
          if (layout != null) {
            for (final tile in _internalTiles) {
              final pos = layout[tile.key];
              if (pos != null) {
                tile.left = _cellLeft(pos.col);
                tile.top = _cellTop(pos.row);
                tile.pixelWidth = _spanWidth(tile.width);
                tile.pixelHeight = _spanHeight(tile.height);
              }
            }
          }
        }

        final rows = _currentRows;
        final totalVerticalSpacing =
            (rows > 0 ? rows - 1 : 0) * widget.mainAxisSpacing;
        final gridHeight = rows * _cellHeight + totalVerticalSpacing;

        Widget gridContent = SizedBox(
          width: constraints.maxWidth,
          height: gridHeight,
          child: Stack(
            clipBehavior: Clip.none,
            children: [
              ..._internalTiles
                  .map((tile) => _buildAnimatedTile(tile)),
            ],
          ),
        );

        if (widget.enableReorder) {
          gridContent = _wrapWithDragListener(gridContent, gridHeight);
        }

        return gridContent;
      },
    );
  }

  Widget _wrapWithDragListener(Widget child, double gridHeight) {
    return DragTarget<Key>(
      onWillAcceptWithDetails: (_) => true,
      onMove: (details) {
        final cell = _cellFromGlobal(details.offset);
        if (cell != null) {
          _onHover(cell);
        }
      },
      onAcceptWithDetails: (details) {
        _handleDrop(details.data, details.offset);
      },
      onLeave: (_) {
        // When leaving the grid area, revert preview
        if (_draggingKey != null && _preDragLayout != null) {
          _hoverPosition = null;
          _lastPreviewPosition = null;
          _applyLayout(_preDragLayout!);
        }
      },
      builder: (context, candidate, rejected) => child,
    );
  }

  Widget _buildAnimatedTile(_InternalTile tile) {
    final tileContent = ClipRRect(
      borderRadius: BorderRadius.circular(widget.borderRadius),
      child: tile.child,
    );

    if (!widget.enableReorder) {
      return AnimatedPositioned(
        duration: widget.animationDuration,
        curve: Curves.easeInOut,
        left: tile.left,
        top: tile.top,
        width: tile.pixelWidth,
        height: tile.pixelHeight,
        child: tileContent,
      );
    }

    final feedbackWidget = SizedBox(
      width: tile.pixelWidth,
      height: tile.pixelHeight,
      child: Transform.scale(
        scale: 1.05,
        child: Material(
          elevation: 8.0,
          color: Colors.transparent,
          borderRadius: BorderRadius.circular(widget.borderRadius),
          child: tileContent,
        ),
      ),
    );

    final placeholder = Container(
      decoration: BoxDecoration(
        color: Theme.of(context).colorScheme.primary.withValues(alpha: 0.08),
        borderRadius: BorderRadius.circular(widget.borderRadius),
        border: Border.all(
          color: Theme.of(context).colorScheme.primary.withValues(alpha: 0.3),
          width: 1.5,
        ),
      ),
    );

    final draggableWidget = LongPressDraggable<Key>(
      data: tile.key,
      onDragStarted: () => _onDragStarted(tile.key),
      onDragEnd: (_) => _onDragEnd(),
      feedback: feedbackWidget,
      childWhenDragging: placeholder,
      child: tileContent,
    );

    final isDragging = _draggingKey == tile.key;
    final justDropped = _justDroppedKey == tile.key;

    // During drag: fade out. After drop: appear instantly (no fade-in).
    // Otherwise: normal opacity with animation.
    final opacity = isDragging ? 0.0 : 1.0;
    final opacityDuration =
        justDropped ? Duration.zero : const Duration(milliseconds: 200);

    return AnimatedPositioned(
      duration: widget.animationDuration,
      curve: Curves.easeInOut,
      left: tile.left,
      top: tile.top,
      width: tile.pixelWidth,
      height: tile.pixelHeight,
      child: AnimatedOpacity(
        duration: opacityDuration,
        opacity: opacity,
        child: draggableWidget,
      ),
    );
  }
}
