import 'package:material_ui/material_ui.dart';

/// A single item of a [ReorderGrid], sized in grid cells.
///
/// This is a plain description, not a widget: the grid reads the spans to place
/// the tile and only then mounts [child].
///
/// ```dart
/// ReorderGridTile.count(
///   key: ValueKey(widget.id),
///   crossAxisCellCount: 2,
///   mainAxisCellCount: 1,
///   child: MyCard(),
/// )
/// ```
@immutable
class ReorderGridTile {
  /// Creates a tile spanning [crossAxisCellCount] columns and
  /// [mainAxisCellCount] rows.
  ///
  /// [key] identifies the tile across rebuilds and reorders and must be unique
  /// within a single grid.
  const ReorderGridTile.count({
    required this.key,
    required this.child,
    this.mainAxisCellCount = 1,
    this.crossAxisCellCount = 1,
    this.borderRadius,
  });

  /// Stable identity of the tile within its grid.
  final Key key;

  /// Span along the main axis, in rows. Values below 1 are treated as 1.
  final int mainAxisCellCount;

  /// Span along the cross axis, in columns.
  ///
  /// Values below 1 are treated as 1; values wider than the grid are narrowed
  /// to the full width instead of breaking the layout.
  final int crossAxisCellCount;

  /// Corner radius for this tile, overriding the grid-wide radius when set.
  final double? borderRadius;

  /// The widget rendered inside the tile's bounds.
  final Widget child;

  @override
  String toString() =>
      'ReorderGridTile($key, ${crossAxisCellCount}x$mainAxisCellCount)';
}
