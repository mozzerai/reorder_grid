import 'package:flutter/scheduler.dart';
import 'package:material_ui/material_ui.dart';

/// Distance from the viewport edge, in logical pixels, inside which a held
/// pointer scrolls the viewport.
const double kEdgeScrollExtent = 48.0;

/// Scroll speed, in logical pixels per second, with the pointer right at the
/// viewport edge. It ramps linearly down to zero at [kEdgeScrollExtent].
const double kEdgeScrollMaxSpeed = 900.0;

/// Scroll left before an extent, in pixels, below which the edge counts as
/// reached.
const double _kExtentTolerance = 1.0;

/// Scrolls a [Scrollable] while a drag holds the pointer near one of its
/// edges.
///
/// It advances once per frame by the time elapsed since the last one, so the
/// content moves the same distance on every frame. Flutter's
/// `EdgeDraggingAutoScroller` chains one short animation per step instead,
/// and each new step only starts on the frame after the last one ends: one
/// frame in three stands still, which reads as stutter.
///
/// The zone follows the pointer rather than the dragged tile: a tile can be
/// taller than the viewport, and a rect that big touches both edges at once.
class EdgeAutoScroller {
  /// Creates a scroller that ticks on [vsync] and calls [onStopped] when a
  /// scroll it started runs out, either because the pointer left the edge
  /// zone or because the content reached its extent.
  EdgeAutoScroller({required TickerProvider vsync, required this.onStopped}) {
    _ticker = vsync.createTicker(_tick);
  }

  /// Called once each time a running scroll comes to rest on its own.
  final VoidCallback onStopped;

  /// The scrollable to drive, or `null` when there is none to drive.
  ScrollableState? scrollable;

  late final Ticker _ticker;
  Offset? _pointer;
  Duration _lastTick = Duration.zero;

  /// Whether a scroll is running.
  bool get isScrolling => _ticker.isActive;

  /// Follows the pointer, starting a scroll if it now pushes against an edge
  /// that still has content beyond it.
  void follow(Offset globalPointer) {
    _pointer = globalPointer;
    if (_ticker.isActive || velocityAt(globalPointer) == 0) return;
    _lastTick = Duration.zero;
    _ticker.start();
  }

  /// Stops any scroll without calling [onStopped], for when the drag ends.
  void stop() {
    _pointer = null;
    if (_ticker.isActive) _ticker.stop();
  }

  /// Releases the ticker.
  void dispose() => _ticker.dispose();

  /// Scroll speed, in pixels per second along the scroll offset, that a
  /// pointer at [globalPointer] asks for; zero when it pushes no edge that
  /// still has content beyond it.
  double velocityAt(Offset globalPointer) {
    final ScrollableState? state = scrollable;
    final RenderObject? viewportBox = state?.context.findRenderObject();
    if (state == null || viewportBox is! RenderBox || !viewportBox.hasSize) {
      return 0;
    }

    final Rect viewport = MatrixUtils.transformRect(
      viewportBox.getTransformTo(null),
      Offset.zero & viewportBox.size,
    );
    final bool vertical =
        axisDirectionToAxis(state.axisDirection) == Axis.vertical;
    final double pointer = vertical ? globalPointer.dy : globalPointer.dx;
    final double start = vertical ? viewport.top : viewport.left;
    final double end = vertical ? viewport.bottom : viewport.right;

    final double towardsEnd = _depthInZone(end - pointer);
    final double towardsStart = _depthInZone(pointer - start);
    final double visual = towardsEnd - towardsStart;
    if (visual == 0) return 0;

    final double offsetDirection = axisDirectionIsReversed(state.axisDirection)
        ? -visual
        : visual;
    final ScrollPosition position = state.position;
    final bool hasRoom = offsetDirection > 0
        ? position.pixels < position.maxScrollExtent - _kExtentTolerance
        : position.pixels > position.minScrollExtent + _kExtentTolerance;
    return hasRoom ? offsetDirection * kEdgeScrollMaxSpeed : 0;
  }

  /// How deep, from 0 to 1, a pointer [distanceToEdge] away from an edge sits
  /// inside that edge's zone.
  static double _depthInZone(double distanceToEdge) =>
      ((kEdgeScrollExtent - distanceToEdge) / kEdgeScrollExtent).clamp(0, 1);

  void _tick(Duration elapsed) {
    final double seconds =
        (elapsed - _lastTick).inMicroseconds / Duration.microsecondsPerSecond;
    _lastTick = elapsed;

    final Offset? pointer = _pointer;
    final ScrollableState? state = scrollable;
    final double velocity = pointer == null ? 0 : velocityAt(pointer);
    if (state == null || velocity == 0) {
      _ticker.stop();
      onStopped();
      return;
    }

    final ScrollPosition position = state.position;
    position.jumpTo(
      (position.pixels + velocity * seconds).clamp(
        position.minScrollExtent,
        position.maxScrollExtent,
      ),
    );
  }
}
