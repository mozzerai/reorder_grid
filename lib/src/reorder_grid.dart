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

/// How long the tile takes to lift off when a drag starts, and to settle back
/// down when it lands.
const Duration _kLiftDuration = Duration(milliseconds: 140);

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
    this.animationDuration = const Duration(milliseconds: 220),
    this.animationCurve = Curves.easeOutCubic,
    this.dragHysteresis = 0.2,
  }) : assert(crossAxisCount > 0, 'crossAxisCount must be greater than zero'),
       assert(cellAspectRatio > 0, 'cellAspectRatio must be greater than zero'),
       assert(
         dragHysteresis >= 0 && dragHysteresis < 0.5,
         'dragHysteresis must be within [0, 0.5)',
       );

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

  /// Deadband, as a fraction of a cell, that the dragged tile must overshoot
  /// before the preview commits to a new slot.
  ///
  /// A tile snaps to the nearest slot, so it changes at the halfway point;
  /// this widens that boundary to stop the preview from flickering when the
  /// pointer rests on it. `0` disables the deadband; must stay below `0.5`.
  final double dragHysteresis;

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

/// Everything a repaint of the grid depends on: which tiles are shown, in which
/// order, in which cell, and which one the pointer is carrying.
@immutable
class _GridSnapshot {
  const _GridSnapshot({
    required this.order,
    required this.layout,
    this.draggingKey,
  });

  static const _GridSnapshot empty = _GridSnapshot(
    order: <Key>[],
    layout: GridLayout.empty,
  );

  /// Visual order of the tiles, which drifts from `widget.children` between a
  /// drop and the parent's rebuild.
  final List<Key> order;

  /// Logical placements. Pixels are derived from these on every build.
  final GridLayout layout;

  /// The tile being dragged, drawn as an empty slot until it lands.
  final Key? draggingKey;
}

/// Holds the current [_GridSnapshot] and rebuilds only the tile stack when it
/// changes, so a drag preview does not rebuild the surrounding drag target.
///
/// Readers must pull [value] inside their builder rather than cache it: a
/// [setQuietly] update deliberately skips the notification.
class _GridSnapshotNotifier extends ChangeNotifier {
  _GridSnapshotNotifier(this._value);

  _GridSnapshot _value;

  _GridSnapshot get value => _value;

  /// Publishes a new snapshot and rebuilds the listeners.
  set value(_GridSnapshot snapshot) {
    if (identical(_value, snapshot)) return;
    _value = snapshot;
    notifyListeners();
  }

  /// Replaces the snapshot without notifying.
  ///
  /// Used from `initState` and `didUpdateWidget`, where the whole grid is about
  /// to be rebuilt anyway and a notification would only mark the stack dirty
  /// twice in the same frame.
  void setQuietly(_GridSnapshot snapshot) => _value = snapshot;
}

class _ReorderGridState extends State<ReorderGrid> {
  final _GridSnapshotNotifier _snapshot = _GridSnapshotNotifier(
    _GridSnapshot.empty,
  );

  Map<Key, ReorderGridTile> _tilesByKey = <Key, ReorderGridTile>{};

  Key? _draggingKey;

  /// Layout to restore when a drag is cancelled or leaves the grid.
  GridLayout? _restingLayout;

  /// Cell the preview is currently pinned to.
  GridPosition? _appliedAnchor;

  List<Key> get _order => _snapshot.value.order;

  GridLayout get _layout => _snapshot.value.layout;

  @override
  void initState() {
    super.initState();
    _adoptChildren();
  }

  @override
  void didUpdateWidget(covariant ReorderGrid oldWidget) {
    super.didUpdateWidget(oldWidget);
    final bool structureChanged =
        widget.crossAxisCount != oldWidget.crossAxisCount ||
        !_sameStructure(oldWidget.children, widget.children);

    if (structureChanged) {
      _adoptChildren();
    } else {
      // Same tiles in the same order: keep our placements (they may be ahead of
      // the parent right after a drop) and just refresh the child widgets. The
      // parent's rebuild already carries the new widgets down to the stack.
      _tilesByKey = _indexByKey(widget.children);
    }
  }

  @override
  void dispose() {
    _snapshot.dispose();
    super.dispose();
  }

  // ── Tile bookkeeping ────────────────────────────────────────────────────

  void _adoptChildren() {
    assert(_debugCheckUniqueKeys(widget.children));
    _tilesByKey = _indexByKey(widget.children);
    final List<Key> order = <Key>[
      for (final ReorderGridTile tile in widget.children) tile.key,
    ];
    // A tile that left the grid mid-drag is no longer being carried.
    final Key? dragging = _snapshot.value.draggingKey;
    _snapshot.setQuietly(
      _GridSnapshot(
        order: order,
        layout: _packOrder(order),
        draggingKey: _tilesByKey.containsKey(dragging) ? dragging : null,
      ),
    );
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
  }) => _packOrder(_order, pinned: pinned);

  GridLayout _packOrder(
    List<Key> order, {
    Map<Key, GridPosition> pinned = const <Key, GridPosition>{},
  }) {
    return packDense(
      columns: widget.crossAxisCount,
      pinned: pinned,
      tiles: <LayoutTile>[
        for (final Key key in order)
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
    _restingLayout = _layout;
    _appliedAnchor = null;
    _draggingKey = key;
    _publish(layout: _layout, draggingKey: key);
    _haptic(HapticFeedback.mediumImpact);
  }

  void _onHover(GridPosition anchor) {
    final Key? key = _draggingKey;
    if (key == null) return;

    final GridPosition clamped = _clampAnchor(anchor, key);
    if (clamped == _appliedAnchor) return;

    _appliedAnchor = clamped;
    _publish(
      layout: _pack(pinned: <Key, GridPosition>{key: clamped}),
      draggingKey: key,
    );
    _haptic(HapticFeedback.selectionClick);
  }

  /// Restores the pre-drag arrangement while keeping the drag alive, used when
  /// the pointer wanders outside the grid.
  void _revertPreview() {
    _appliedAnchor = null;
    final GridLayout? resting = _restingLayout;
    if (resting != null) _publish(layout: resting, draggingKey: _draggingKey);
  }

  /// Ends the drag. Fired by `Draggable.onDragEnd`, which runs after a
  /// successful drop, so [accepted] tells us whether [_handleDrop] already
  /// committed a new arrangement.
  void _endDrag({required bool accepted}) {
    if (_draggingKey == null && _restingLayout == null) return;

    final GridLayout? resting = _restingLayout;
    _publish(layout: !accepted && resting != null ? resting : _layout);
    _draggingKey = null;
    _restingLayout = null;
    _appliedAnchor = null;
  }

  void _handleDrop(Key key, Offset globalOffset, GridGeometry geometry) {
    final Offset? releasedAt = _toLocal(globalOffset);
    final GridPosition? target =
        _appliedAnchor ??
        (releasedAt == null ? null : _anchorAt(releasedAt, geometry));
    if (target == null) {
      _endDrag(accepted: false);
      return;
    }

    final GridPosition landing = _clampAnchor(target, key);
    final GridLayout layout = _pack(pinned: <Key, GridPosition>{key: landing});

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

    _publish(order: newOrder, layout: layout);
    _draggingKey = null;
    _restingLayout = null;
    _appliedAnchor = null;

    if (oldIndex >= 0 && newIndex >= 0 && oldIndex != newIndex) {
      widget.onReorder?.call(oldIndex, newIndex);
    }
  }

  /// Publishes a new snapshot, rebuilding only the tile stack.
  void _publish({
    List<Key>? order,
    required GridLayout layout,
    Key? draggingKey,
  }) {
    final _GridSnapshot current = _snapshot.value;
    if (order == null &&
        draggingKey == current.draggingKey &&
        identical(current.layout, layout)) {
      return;
    }
    _snapshot.value = _GridSnapshot(
      order: order ?? current.order,
      layout: layout,
      draggingKey: draggingKey,
    );
  }

  Offset? _toLocal(Offset globalOffset) {
    final RenderBox? box = context.findRenderObject() as RenderBox?;
    if (box == null || !box.hasSize) return null;
    return box.globalToLocal(globalOffset);
  }

  /// Snaps the dragged tile's top-left to the slot it should take.
  GridPosition? _anchorAt(Offset local, GridGeometry geometry) =>
      geometry.snapAnchor(
        local,
        current: _appliedAnchor,
        hysteresis: widget.dragHysteresis,
      );

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

        // Only the stack listens to the snapshot, so a drag preview leaves the
        // drag target and the layout builder untouched. The snapshot is read
        // here, inside the builder, so a quiet update still lands on the next
        // rebuild of the grid.
        final Widget grid = ListenableBuilder(
          listenable: _snapshot,
          builder: (BuildContext context, Widget? child) {
            final _GridSnapshot snapshot = _snapshot.value;
            return SizedBox(
              width: constraints.maxWidth,
              height: geometry.totalHeight(snapshot.layout.rows),
              child: Stack(
                clipBehavior: Clip.none,
                children: <Widget>[
                  if (widget.showSlotBorders)
                    _buildSlotBorders(context, geometry, snapshot),
                  for (final Key key in snapshot.order)
                    _buildTile(context, key, geometry, snapshot),
                ],
              ),
            );
          },
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
        final Offset? local = _toLocal(details.offset);
        if (local == null) return;
        final GridPosition? anchor = _anchorAt(local, geometry);
        if (anchor != null) _onHover(anchor);
      },
      onAcceptWithDetails: (DragTargetDetails<_ReorderDragData> details) =>
          _handleDrop(details.data.tileKey, details.offset, geometry),
      onLeave: (_) => _revertPreview(),
      builder: (BuildContext context, _, _) => child,
    );
  }

  Widget _buildTile(
    BuildContext context,
    Key key,
    GridGeometry geometry,
    _GridSnapshot snapshot,
  ) {
    final ReorderGridTile tile = _tilesByKey[key]!;
    final GridPosition position = snapshot.layout[key] ?? kGridOrigin;
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
              // `childWhenDragging` is deliberately unset: swapping the child
              // out unmounts its subtree, and remounting it on drop restarts
              // whatever state it holds — async gates re-enter their pending
              // branch, animations replay. The tile stays mounted and is
              // hidden in place instead.
              child: _hideWhileDragging(
                context: context,
                content: content,
                radius: radius,
                dragging: snapshot.draggingKey == key,
              ),
            )
          : content,
    );
  }

  /// Shows the drop placeholder in the tile's slot while its content stays
  /// mounted, laid out and invisible behind it.
  ///
  /// The tree shape is the same whether or not the tile is being dragged — two
  /// children, in the same order, of the same types. Anything else would let
  /// the framework discard the content's element on the toggle, which is the
  /// remount this exists to avoid.
  Widget _hideWhileDragging({
    required BuildContext context,
    required Widget content,
    required BorderRadius radius,
    required bool dragging,
  }) {
    return Stack(
      fit: StackFit.expand,
      children: <Widget>[
        Visibility(
          visible: dragging,
          child: _buildDropPlaceholder(context, radius),
        ),
        Visibility(
          visible: !dragging,
          maintainState: true,
          maintainAnimation: true,
          maintainSize: true,
          child: content,
        ),
      ],
    );
  }

  /// The tile floating under the pointer.
  ///
  /// It eases into its raised state instead of appearing already lifted, which
  /// is what signals "you picked this up" rather than "this blinked".
  Widget _buildFeedback(
    Widget content,
    double width,
    double height,
    BorderRadius radius,
  ) {
    return SizedBox(
      width: width,
      height: height,
      child: TweenAnimationBuilder<double>(
        tween: Tween<double>(begin: 0.0, end: 1.0),
        duration: _kLiftDuration,
        curve: Curves.easeOutCubic,
        builder: (BuildContext context, double lift, Widget? child) {
          return Transform.scale(
            scale: 1.0 + (_kFeedbackScale - 1.0) * lift,
            child: Material(
              elevation: _kFeedbackElevation * lift,
              color: Colors.transparent,
              borderRadius: radius,
              child: child,
            ),
          );
        },
        child: content,
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

  Widget _buildSlotBorders(
    BuildContext context,
    GridGeometry geometry,
    _GridSnapshot snapshot,
  ) {
    final Set<int> occupied = <int>{};
    for (final Key key in snapshot.order) {
      final ReorderGridTile? tile = _tilesByKey[key];
      final GridPosition? position = snapshot.layout[key];
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
            rows: snapshot.layout.rows,
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
