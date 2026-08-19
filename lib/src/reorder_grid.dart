import 'dart:async';
import 'dart:math' as math;

import 'package:flutter/foundation.dart';
import 'package:flutter/services.dart';
import 'package:material_ui/material_ui.dart';

import 'dense_layout.dart';
import 'grid_geometry.dart';
import 'grid_position.dart';
import 'reorder_grid_tile.dart';

/// Scale applied to the tile floating under the pointer while dragging.
const double _kFeedbackScale = 1.05;

/// Elevation of the floating tile while dragging.
const double _kFeedbackElevation = 8.0;

/// Signature of the callback invoked once a tile lands in a new slot.
///
/// Indices refer to positions in the `children` list, in reading order.
typedef ReorderGridCallback = void Function(int oldIndex, int newIndex);

/// A grid of variable-sized tiles that the user can reorder by dragging.
///
/// Tiles are packed densely: each one takes the topmost, leftmost slot it fits
/// into, so wide and tall tiles coexist without leaving holes. Positions are
/// kept as grid cells and converted to pixels on every build, which means the
/// grid reflows on its own when the available width or the spacing changes.
///
/// While a drag is in progress the grid shows a live preview: the dragged tile
/// is pinned under the pointer and the remaining tiles flow around it. Dropping
/// commits the preview and reports the movement through [onReorder]; releasing
/// outside the grid restores the previous arrangement.
///
/// ```dart
/// ReorderGrid.count(
///   crossAxisCount: 4,
///   onReorder: viewModel.reorder,
///   children: [
///     for (final item in items)
///       ReorderGridTile.count(
///         key: ValueKey(item.id),
///         crossAxisCellCount: item.width,
///         mainAxisCellCount: item.height,
///         child: ItemCard(item),
///       ),
///   ],
/// )
/// ```
///
/// The grid sizes itself to its content along the main axis and requires a
/// bounded width, so it is meant to be placed inside a scrollable rather than
/// to scroll on its own.
class ReorderGrid extends StatefulWidget {
  /// Creates a grid with a fixed number of [crossAxisCount] columns.
  const ReorderGrid.count({
    super.key,
    required this.crossAxisCount,
    required this.children,
    this.mainAxisSpacing = 8.0,
    this.crossAxisSpacing = 8.0,
    this.cellAspectRatio = 1.0,
    this.enableReorder = true,
    this.enableHapticFeedback = true,
    this.showSlotBorders = false,
    this.slotBorderColor,
    this.onReorder,
    this.borderRadius = 8.0,
    this.animationDuration = const Duration(milliseconds: 300),
    this.animationCurve = Curves.easeInOut,
    this.previewDelay = const Duration(milliseconds: 150),
  }) : assert(crossAxisCount > 0, 'crossAxisCount must be greater than zero'),
       assert(cellAspectRatio > 0, 'cellAspectRatio must be greater than zero');

  /// Number of columns the grid is divided into.
  final int crossAxisCount;

  /// The tiles to lay out, in reading order.
  ///
  /// Keys must be unique; duplicates throw in debug builds.
  final List<ReorderGridTile> children;

  /// Gap between rows, in logical pixels.
  final double mainAxisSpacing;

  /// Gap between columns, in logical pixels.
  final double crossAxisSpacing;

  /// Width divided by height of a single cell, as in
  /// `GridView.childAspectRatio`. Defaults to square cells.
  final double cellAspectRatio;

  /// Whether tiles can be dragged. When `false` the grid is a plain dense
  /// layout with no drag machinery attached.
  final bool enableReorder;

  /// Whether to fire haptics on drag start and on every preview change.
  final bool enableHapticFeedback;

  /// Whether to outline the cells no tile occupies.
  final bool showSlotBorders;

  /// Colour of the empty-slot outlines. Defaults to the theme's
  /// `colorScheme.outlineVariant`.
  final Color? slotBorderColor;

  /// Called after a drop that changed the tile's position.
  final ReorderGridCallback? onReorder;

  /// Corner radius applied to tiles that do not define their own.
  final double borderRadius;

  /// Duration of the reflow animation when tiles change slot.
  final Duration animationDuration;

  /// Curve of the reflow animation.
  final Curve animationCurve;

  /// How long the pointer must rest over a new cell before the preview
  /// reflows. Adds hysteresis so large tiles do not thrash between slots.
  final Duration previewDelay;

  @override
  State<ReorderGrid> createState() => _ReorderGridState();
}

/// Payload carried by a drag, scoped to the grid that started it so two grids
/// on screen never steal each other's tiles.
@immutable
class _ReorderDragData {
  const _ReorderDragData({required this.owner, required this.tileKey});

  final Object owner;
  final Key tileKey;
}

class _ReorderGridState extends State<ReorderGrid> {
  /// Visual order of the tiles, which drifts from `widget.children` between a
  /// drop and the parent's rebuild.
  final List<Key> _order = <Key>[];

  Map<Key, ReorderGridTile> _tilesByKey = <Key, ReorderGridTile>{};

  /// Logical placements. Pixels are derived from these on every build.
  GridLayout _layout = GridLayout.empty;

  Key? _draggingKey;

  /// Layout to restore when a drag is cancelled or leaves the grid.
  GridLayout? _restingLayout;

  /// Cell the preview is currently pinned to.
  GridPosition? _appliedAnchor;

  /// Cell waiting out [ReorderGrid.previewDelay] before becoming the preview.
  GridPosition? _pendingAnchor;

  Timer? _previewTimer;

  @override
  void initState() {
    super.initState();
    _adoptChildren();
    _layout = _pack();
  }

  @override
  void didUpdateWidget(covariant ReorderGrid oldWidget) {
    super.didUpdateWidget(oldWidget);
    final bool structureChanged =
        widget.crossAxisCount != oldWidget.crossAxisCount ||
        !_sameStructure(oldWidget.children, widget.children);

    if (structureChanged) {
      // A build is already scheduled, so mutate directly instead of setState.
      _adoptChildren();
      _layout = _pack();
    } else {
      // Same tiles in the same order: keep our placements (they may be ahead of
      // the parent right after a drop) and just refresh the child widgets.
      _tilesByKey = _indexByKey(widget.children);
    }
  }

  @override
  void dispose() {
    _previewTimer?.cancel();
    super.dispose();
  }

  // ── Tile bookkeeping ────────────────────────────────────────────────────

  void _adoptChildren() {
    assert(_debugCheckUniqueKeys(widget.children));
    _order
      ..clear()
      ..addAll(widget.children.map((ReorderGridTile tile) => tile.key));
    _tilesByKey = _indexByKey(widget.children);
  }

  static Map<Key, ReorderGridTile> _indexByKey(List<ReorderGridTile> tiles) => {
    for (final ReorderGridTile tile in tiles) tile.key: tile,
  };

  /// Whether two child lists describe the same grid, ignoring the widgets
  /// themselves. Only identity and spans can invalidate a layout.
  static bool _sameStructure(List<ReorderGridTile> a, List<ReorderGridTile> b) {
    if (identical(a, b)) return true;
    if (a.length != b.length) return false;
    for (int i = 0; i < a.length; i++) {
      if (a[i].key != b[i].key ||
          a[i].crossAxisCellCount != b[i].crossAxisCellCount ||
          a[i].mainAxisCellCount != b[i].mainAxisCellCount) {
        return false;
      }
    }
    return true;
  }

  static bool _debugCheckUniqueKeys(List<ReorderGridTile> tiles) {
    final Set<Key> seen = <Key>{};
    for (final ReorderGridTile tile in tiles) {
      if (!seen.add(tile.key)) {
        throw FlutterError.fromParts(<DiagnosticsNode>[
          ErrorSummary('Duplicate key in ReorderGrid children: ${tile.key}.'),
          ErrorDescription(
            'Every ReorderGridTile needs a key that is unique within its grid, '
            'otherwise tiles overwrite each other and disappear.',
          ),
        ]);
      }
    }
    return true;
  }

  int _columnSpan(ReorderGridTile tile) =>
      tile.crossAxisCellCount.clamp(1, widget.crossAxisCount);

  int _rowSpan(ReorderGridTile tile) => math.max(1, tile.mainAxisCellCount);

  // ── Layout ──────────────────────────────────────────────────────────────

  GridLayout _pack({
    Map<Key, GridPosition> pinned = const <Key, GridPosition>{},
  }) {
    return packDense(
      columns: widget.crossAxisCount,
      pinned: pinned,
      tiles: <LayoutTile>[
        for (final Key key in _order)
          if (_tilesByKey[key] case final ReorderGridTile tile)
            LayoutTile(
              key: key,
              width: _columnSpan(tile),
              height: _rowSpan(tile),
            ),
      ],
    );
  }

  /// Keeps [position] inside the grid for the tile identified by [key].
  GridPosition _clampAnchor(GridPosition position, Key key) {
    final ReorderGridTile? tile = _tilesByKey[key];
    final int width = tile == null ? 1 : _columnSpan(tile);
    final int maxCol = math.max(0, widget.crossAxisCount - width);
    return (row: math.max(0, position.row), col: position.col.clamp(0, maxCol));
  }

  // ── Drag lifecycle ──────────────────────────────────────────────────────

  void _onDragStarted(Key key) {
    _previewTimer?.cancel();
    _restingLayout = _layout;
    _appliedAnchor = null;
    _pendingAnchor = null;
    setState(() => _draggingKey = key);
    _haptic(HapticFeedback.mediumImpact);
  }

  void _onHover(GridPosition position) {
    final Key? key = _draggingKey;
    if (key == null) return;

    final GridPosition anchor = _clampAnchor(position, key);
    if (anchor == _appliedAnchor) {
      // Back where the preview already is: drop any pending move.
      _previewTimer?.cancel();
      _pendingAnchor = null;
      return;
    }
    if (anchor == _pendingAnchor) return;

    _pendingAnchor = anchor;
    _previewTimer?.cancel();
    _previewTimer = Timer(widget.previewDelay, () {
      if (!mounted || _draggingKey != key) return;
      _appliedAnchor = anchor;
      _pendingAnchor = null;
      setState(() => _layout = _pack(pinned: <Key, GridPosition>{key: anchor}));
      _haptic(HapticFeedback.selectionClick);
    });
  }

  /// Restores the pre-drag arrangement while keeping the drag alive, used when
  /// the pointer wanders outside the grid.
  void _revertPreview() {
    _previewTimer?.cancel();
    _pendingAnchor = null;
    _appliedAnchor = null;
    final GridLayout? resting = _restingLayout;
    if (resting != null && !identical(resting, _layout)) {
      setState(() => _layout = resting);
    }
  }

  /// Ends the drag. Fired by `Draggable.onDragEnd`, which runs after a
  /// successful drop, so [accepted] tells us whether [_handleDrop] already
  /// committed a new arrangement.
  void _endDrag({required bool accepted}) {
    _previewTimer?.cancel();
    if (_draggingKey == null && _restingLayout == null) return;

    final GridLayout? resting = _restingLayout;
    setState(() {
      if (!accepted && resting != null) _layout = resting;
      _draggingKey = null;
    });
    _restingLayout = null;
    _appliedAnchor = null;
    _pendingAnchor = null;
  }

  void _handleDrop(Key key, Offset globalOffset, GridGeometry geometry) {
    _previewTimer?.cancel();

    final GridPosition? target =
        _appliedAnchor ??
        _pendingAnchor ??
        _cellFromGlobal(globalOffset, geometry);
    if (target == null) {
      _endDrag(accepted: false);
      return;
    }

    final GridLayout layout = _pack(
      pinned: <Key, GridPosition>{key: _clampAnchor(target, key)},
    );

    final List<MapEntry<Key, GridPosition>> readingOrder =
        layout.placements.entries.toList()..sort(
          (MapEntry<Key, GridPosition> a, MapEntry<Key, GridPosition> b) =>
              compareRowMajor(a.value, b.value),
        );
    final List<Key> newOrder = <Key>[
      for (final MapEntry<Key, GridPosition> entry in readingOrder) entry.key,
    ];

    final int oldIndex = widget.children.indexWhere(
      (ReorderGridTile tile) => tile.key == key,
    );
    final int newIndex = newOrder.indexOf(key);

    setState(() {
      _layout = layout;
      _order
        ..clear()
        ..addAll(newOrder);
      _draggingKey = null;
    });
    _restingLayout = null;
    _appliedAnchor = null;
    _pendingAnchor = null;

    if (oldIndex >= 0 && newIndex >= 0 && oldIndex != newIndex) {
      widget.onReorder?.call(oldIndex, newIndex);
    }
  }

  GridPosition? _cellFromGlobal(Offset globalOffset, GridGeometry geometry) {
    final RenderBox? box = context.findRenderObject() as RenderBox?;
    if (box == null || !box.hasSize) return null;
    // One row past the bottom stays addressable so a tile can be dropped into a
    // brand-new last row.
    return geometry.cellAt(
      box.globalToLocal(globalOffset),
      rows: _layout.rows + 1,
    );
  }

  void _haptic(Future<void> Function() impact) {
    if (widget.enableHapticFeedback) unawaited(impact());
  }

  // ── Build ───────────────────────────────────────────────────────────────

  @override
  Widget build(BuildContext context) {
    return LayoutBuilder(
      builder: (BuildContext context, BoxConstraints constraints) {
        assert(
          constraints.hasBoundedWidth,
          'ReorderGrid requires a bounded width. Wrap it in a widget that '
          'constrains its width, such as SizedBox or Expanded.',
        );
        if (!constraints.hasBoundedWidth) return const SizedBox.shrink();

        final GridGeometry geometry = GridGeometry.fromWidth(
          availableWidth: constraints.maxWidth,
          columns: widget.crossAxisCount,
          mainAxisSpacing: widget.mainAxisSpacing,
          crossAxisSpacing: widget.crossAxisSpacing,
          cellAspectRatio: widget.cellAspectRatio,
        );

        final Widget grid = SizedBox(
          width: constraints.maxWidth,
          height: geometry.totalHeight(_layout.rows),
          child: Stack(
            clipBehavior: Clip.none,
            children: <Widget>[
              if (widget.showSlotBorders) _buildSlotBorders(context, geometry),
              for (final Key key in _order) _buildTile(context, key, geometry),
            ],
          ),
        );

        if (!widget.enableReorder) return grid;
        return _buildDragTarget(grid, geometry);
      },
    );
  }

  Widget _buildDragTarget(Widget child, GridGeometry geometry) {
    return DragTarget<_ReorderDragData>(
      onWillAcceptWithDetails: (DragTargetDetails<_ReorderDragData> details) =>
          identical(details.data.owner, this),
      onMove: (DragTargetDetails<_ReorderDragData> details) {
        final GridPosition? cell = _cellFromGlobal(details.offset, geometry);
        if (cell != null) _onHover(cell);
      },
      onAcceptWithDetails: (DragTargetDetails<_ReorderDragData> details) =>
          _handleDrop(details.data.tileKey, details.offset, geometry),
      onLeave: (_) => _revertPreview(),
      builder: (BuildContext context, _, _) => child,
    );
  }

  Widget _buildTile(BuildContext context, Key key, GridGeometry geometry) {
    final ReorderGridTile tile = _tilesByKey[key]!;
    final GridPosition position = _layout[key] ?? kGridOrigin;
    final BorderRadius radius = BorderRadius.circular(
      tile.borderRadius ?? widget.borderRadius,
    );
    final double width = geometry.widthOf(_columnSpan(tile));
    final double height = geometry.heightOf(_rowSpan(tile));

    final Widget content = ClipRRect(borderRadius: radius, child: tile.child);

    return AnimatedPositioned(
      // Keyed so the implicit animation follows the tile across reorders
      // instead of following its index in the stack.
      key: key,
      duration: widget.animationDuration,
      curve: widget.animationCurve,
      left: geometry.leftOf(position.col),
      top: geometry.topOf(position.row),
      width: width,
      height: height,
      child: widget.enableReorder
          ? LongPressDraggable<_ReorderDragData>(
              data: _ReorderDragData(owner: this, tileKey: key),
              onDragStarted: () => _onDragStarted(key),
              onDragEnd: (DraggableDetails details) =>
                  _endDrag(accepted: details.wasAccepted),
              feedback: _buildFeedback(content, width, height, radius),
              childWhenDragging: _buildDropPlaceholder(context, radius),
              child: content,
            )
          : content,
    );
  }

  Widget _buildFeedback(
    Widget content,
    double width,
    double height,
    BorderRadius radius,
  ) {
    return SizedBox(
      width: width,
      height: height,
      child: Transform.scale(
        scale: _kFeedbackScale,
        child: Material(
          elevation: _kFeedbackElevation,
          color: Colors.transparent,
          borderRadius: radius,
          child: content,
        ),
      ),
    );
  }

  Widget _buildDropPlaceholder(BuildContext context, BorderRadius radius) {
    final Color accent = Theme.of(context).colorScheme.primary;
    return DecoratedBox(
      decoration: BoxDecoration(
        color: accent.withValues(alpha: 0.08),
        borderRadius: radius,
        border: Border.all(color: accent.withValues(alpha: 0.3), width: 1.5),
      ),
    );
  }

  Widget _buildSlotBorders(BuildContext context, GridGeometry geometry) {
    final Set<int> occupied = <int>{};
    for (final Key key in _order) {
      final ReorderGridTile? tile = _tilesByKey[key];
      final GridPosition? position = _layout[key];
      if (tile == null || position == null) continue;
      final int width = _columnSpan(tile);
      final int height = _rowSpan(tile);
      for (int row = position.row; row < position.row + height; row++) {
        for (int col = position.col; col < position.col + width; col++) {
          occupied.add(row * widget.crossAxisCount + col);
        }
      }
    }

    return Positioned.fill(
      child: IgnorePointer(
        child: CustomPaint(
          painter: _SlotBorderPainter(
            geometry: geometry,
            columns: widget.crossAxisCount,
            rows: _layout.rows,
            occupied: occupied,
            color:
                widget.slotBorderColor ??
                Theme.of(context).colorScheme.outlineVariant,
            radius: widget.borderRadius,
          ),
        ),
      ),
    );
  }
}

/// Outlines the cells no tile occupies, giving the user a sense of where a
/// dragged tile can land.
class _SlotBorderPainter extends CustomPainter {
  const _SlotBorderPainter({
    required this.geometry,
    required this.columns,
    required this.rows,
    required this.occupied,
    required this.color,
    required this.radius,
  });

  final GridGeometry geometry;
  final int columns;
  final int rows;
  final Set<int> occupied;
  final Color color;
  final double radius;

  @override
  void paint(Canvas canvas, Size size) {
    if (rows <= 0 || geometry.cellWidth <= 0) return;
    final Paint paint = Paint()
      ..style = PaintingStyle.stroke
      ..strokeWidth = 1.0
      ..color = color;
    final Radius corner = Radius.circular(radius);

    for (int row = 0; row < rows; row++) {
      for (int col = 0; col < columns; col++) {
        if (occupied.contains(row * columns + col)) continue;
        canvas.drawRRect(
          RRect.fromRectAndRadius(
            Rect.fromLTWH(
              geometry.leftOf(col),
              geometry.topOf(row),
              geometry.cellWidth,
              geometry.cellHeight,
            ),
            corner,
          ),
          paint,
        );
      }
    }
  }

  @override
  bool shouldRepaint(_SlotBorderPainter oldDelegate) =>
      geometry != oldDelegate.geometry ||
      columns != oldDelegate.columns ||
      rows != oldDelegate.rows ||
      color != oldDelegate.color ||
      radius != oldDelegate.radius ||
      !setEquals(occupied, oldDelegate.occupied);
}
