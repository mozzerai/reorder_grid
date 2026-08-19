import 'package:flutter_test/flutter_test.dart';
import 'package:material_ui/material_ui.dart';
import 'package:reorder_grid/reorder_grid.dart';

const Duration _pastLongPress = Duration(milliseconds: 600);
const Duration _pastPreview = Duration(milliseconds: 200);

ReorderGridTile box(String id, {int width = 1, int height = 1}) {
  return ReorderGridTile.count(
    key: ValueKey<String>(id),
    crossAxisCellCount: width,
    mainAxisCellCount: height,
    child: ColoredBox(
      key: ValueKey<String>('box-$id'),
      color: const Color(0xFF112233),
      child: const SizedBox.expand(),
    ),
  );
}

Finder boxFinder(String id) => find.byKey(ValueKey<String>('box-$id'));

/// Counts how many times a tile's content is mounted and torn down.
class _MountLog {
  int mounts = 0;
  int disposals = 0;
}

class _LifecycleProbe extends StatefulWidget {
  const _LifecycleProbe({required this.log});

  final _MountLog log;

  @override
  State<_LifecycleProbe> createState() => _LifecycleProbeState();
}

class _LifecycleProbeState extends State<_LifecycleProbe> {
  @override
  void initState() {
    super.initState();
    widget.log.mounts++;
  }

  @override
  void dispose() {
    widget.log.disposals++;
    super.dispose();
  }

  @override
  Widget build(BuildContext context) => ColoredBox(
    key: const ValueKey<String>('box-d'),
    color: const Color(0xFF112233),
    child: const SizedBox.expand(),
  );
}

Widget host({
  required List<ReorderGridTile> children,
  double width = 400,
  int columns = 4,
  double spacing = 0,
  bool enableReorder = true,
  bool showSlotBorders = false,
  Duration duration = const Duration(milliseconds: 220),
  ReorderGridCallback? onReorder,
}) {
  return MaterialApp(
    home: Scaffold(
      body: Align(
        alignment: Alignment.topLeft,
        child: SizedBox(
          width: width,
          child: ReorderGrid.count(
            crossAxisCount: columns,
            mainAxisSpacing: spacing,
            crossAxisSpacing: spacing,
            enableReorder: enableReorder,
            enableHapticFeedback: false,
            showSlotBorders: showSlotBorders,
            animationDuration: duration,
            onReorder: onReorder,
            children: children,
          ),
        ),
      ),
    ),
  );
}

/// Starts a long-press drag on the tile [id] and returns the live gesture.
Future<TestGesture> beginDrag(WidgetTester tester, String id) async {
  final TestGesture gesture = await tester.startGesture(
    tester.getCenter(boxFinder(id)),
  );
  await tester.pump(_pastLongPress);
  await tester.pump();
  return gesture;
}

void main() {
  group('layout', () {
    testWidgets('places tiles densely in reading order', (
      WidgetTester tester,
    ) async {
      await tester.pumpWidget(
        host(
          columns: 4,
          children: <ReorderGridTile>[
            box('tall', height: 2),
            box('wide', width: 3),
            box('small'),
          ],
        ),
      );

      expect(tester.getTopLeft(boxFinder('tall')), Offset.zero);
      expect(tester.getSize(boxFinder('tall')), const Size(100, 200));

      expect(tester.getTopLeft(boxFinder('wide')), const Offset(100, 0));
      expect(tester.getSize(boxFinder('wide')), const Size(300, 100));

      // Backfills the hole under the wide tile.
      expect(tester.getTopLeft(boxFinder('small')), const Offset(100, 100));
    });

    testWidgets('sizes itself to the packed row count', (
      WidgetTester tester,
    ) async {
      await tester.pumpWidget(
        host(
          columns: 2,
          spacing: 10,
          width: 210,
          children: <ReorderGridTile>[box('a'), box('b'), box('c')],
        ),
      );

      // cell = (210 - 10) / 2 = 100; two rows with one inner gap.
      expect(tester.getSize(find.byType(ReorderGrid)).height, 210);
    });

    testWidgets('honours the tile spans against a narrower grid', (
      WidgetTester tester,
    ) async {
      await tester.pumpWidget(
        host(columns: 2, children: <ReorderGridTile>[box('huge', width: 8)]),
      );

      expect(tester.getSize(boxFinder('huge')), const Size(400, 200));
    });

    testWidgets('reflows when the available width changes', (
      WidgetTester tester,
    ) async {
      final List<ReorderGridTile> children = <ReorderGridTile>[
        box('a'),
        box('b'),
      ];

      await tester.pumpWidget(host(children: children));
      expect(tester.getSize(boxFinder('a')), const Size(100, 100));

      await tester.pumpWidget(host(children: children, width: 200));
      await tester.pumpAndSettle();

      expect(tester.getSize(boxFinder('a')), const Size(50, 50));
      expect(tester.getTopLeft(boxFinder('b')), const Offset(50, 0));
    });

    testWidgets('reflows when the column count changes', (
      WidgetTester tester,
    ) async {
      final List<ReorderGridTile> children = <ReorderGridTile>[
        box('a'),
        box('b'),
      ];

      await tester.pumpWidget(host(children: children, columns: 2));
      expect(tester.getTopLeft(boxFinder('b')), const Offset(200, 0));

      await tester.pumpWidget(host(children: children, columns: 1));
      await tester.pumpAndSettle();

      expect(tester.getTopLeft(boxFinder('b')), const Offset(0, 400));
    });

    testWidgets('rejects duplicate keys', (WidgetTester tester) async {
      await tester.pumpWidget(
        host(children: <ReorderGridTile>[box('a'), box('a')]),
      );

      expect(tester.takeException(), isFlutterError);
    });

    testWidgets('paints slot borders only when asked', (
      WidgetTester tester,
    ) async {
      final Finder slotBorders = find.byWidgetPredicate(
        (Widget widget) =>
            widget is CustomPaint &&
            widget.painter.runtimeType.toString() == '_SlotBorderPainter',
      );

      await tester.pumpWidget(
        host(columns: 2, children: <ReorderGridTile>[box('a')]),
      );
      expect(slotBorders, findsNothing);

      await tester.pumpWidget(
        host(
          columns: 2,
          showSlotBorders: true,
          children: <ReorderGridTile>[box('a')],
        ),
      );
      expect(slotBorders, findsOneWidget);
    });
  });

  group('reordering', () {
    testWidgets('reports the reading-order indices after a drop', (
      WidgetTester tester,
    ) async {
      int? oldIndex;
      int? newIndex;

      await tester.pumpWidget(
        host(
          columns: 2,
          width: 200,
          onReorder: (int from, int to) {
            oldIndex = from;
            newIndex = to;
          },
          children: <ReorderGridTile>[box('a'), box('b'), box('c'), box('d')],
        ),
      );

      final TestGesture gesture = await tester.startGesture(
        tester.getCenter(boxFinder('d')),
      );
      await tester.pump(_pastLongPress);
      await tester.pump();

      await gesture.moveTo(const Offset(50, 50));
      await tester.pump(_pastPreview);

      await gesture.up();
      await tester.pumpAndSettle();

      expect(oldIndex, 3);
      expect(newIndex, 0);
    });

    testWidgets('previews the new arrangement before the drop', (
      WidgetTester tester,
    ) async {
      await tester.pumpWidget(
        host(
          columns: 2,
          width: 200,
          children: <ReorderGridTile>[box('a'), box('b'), box('c'), box('d')],
        ),
      );

      final TestGesture gesture = await tester.startGesture(
        tester.getCenter(boxFinder('d')),
      );
      await tester.pump(_pastLongPress);
      await tester.pump();

      await gesture.moveTo(const Offset(50, 50));
      await tester.pump(_pastPreview);
      await tester.pumpAndSettle();

      // 'a' has been pushed aside to make room for the dragged tile.
      expect(tester.getTopLeft(boxFinder('a')), const Offset(100, 0));

      await gesture.up();
      await tester.pumpAndSettle();
    });

    testWidgets('restores the layout when the drag is cancelled', (
      WidgetTester tester,
    ) async {
      int reorderCalls = 0;

      await tester.pumpWidget(
        host(
          columns: 2,
          width: 200,
          onReorder: (int from, int to) => reorderCalls++,
          children: <ReorderGridTile>[box('a'), box('b'), box('c'), box('d')],
        ),
      );

      final TestGesture gesture = await tester.startGesture(
        tester.getCenter(boxFinder('d')),
      );
      await tester.pump(_pastLongPress);
      await tester.pump();

      await gesture.moveTo(const Offset(50, 50));
      await tester.pump(_pastPreview);
      await gesture.moveTo(const Offset(600, 500));
      await tester.pump(_pastPreview);

      await gesture.up();
      await tester.pumpAndSettle();

      expect(reorderCalls, 0);
      expect(tester.getTopLeft(boxFinder('a')), Offset.zero);
      expect(tester.getTopLeft(boxFinder('d')), const Offset(100, 100));
    });

    testWidgets('keeps its own order until the parent catches up', (
      WidgetTester tester,
    ) async {
      await tester.pumpWidget(
        host(
          columns: 2,
          width: 200,
          onReorder: (int from, int to) {},
          children: <ReorderGridTile>[box('a'), box('b'), box('c'), box('d')],
        ),
      );

      final TestGesture gesture = await tester.startGesture(
        tester.getCenter(boxFinder('d')),
      );
      await tester.pump(_pastLongPress);
      await tester.pump();
      await gesture.moveTo(const Offset(50, 50));
      await tester.pump(_pastPreview);
      await gesture.up();
      await tester.pumpAndSettle();

      // The parent rebuilds with the old order because it ignored onReorder.
      await tester.pumpWidget(
        host(
          columns: 2,
          width: 200,
          onReorder: (int from, int to) {},
          children: <ReorderGridTile>[box('a'), box('b'), box('c'), box('d')],
        ),
      );
      await tester.pumpAndSettle();

      expect(tester.getTopLeft(boxFinder('d')), Offset.zero);
    });

    testWidgets('previews on the very next frame, with no dwell time', (
      WidgetTester tester,
    ) async {
      await tester.pumpWidget(
        host(
          columns: 2,
          width: 200,
          duration: Duration.zero,
          children: <ReorderGridTile>[box('a'), box('b'), box('c'), box('d')],
        ),
      );

      final TestGesture gesture = await beginDrag(tester, 'd');
      await gesture.moveTo(const Offset(50, 50));
      await tester.pump();

      // One frame is enough: nothing waits on a timer.
      expect(tester.getTopLeft(boxFinder('a')), const Offset(100, 0));

      await gesture.up();
      await tester.pumpAndSettle();
    });

    testWidgets('snaps to the nearest slot, holding a deadband', (
      WidgetTester tester,
    ) async {
      await tester.pumpWidget(
        host(
          columns: 2,
          width: 200,
          duration: Duration.zero,
          children: <ReorderGridTile>[box('a'), box('b'), box('c'), box('d')],
        ),
      );

      // 'd' sits bottom-right; 'c' bottom-left marks where the preview is.
      expect(tester.getTopLeft(boxFinder('c')), const Offset(0, 100));

      // Pressing the centre of 'd' anchors the drag 50px into the tile.
      final TestGesture gesture = await beginDrag(tester, 'd');

      // Tile top-left lands at x=40, i.e. 0.4 of a cell — past the midpoint
      // of nothing yet, so the anchor stays put.
      await gesture.moveTo(const Offset(90, 150));
      await tester.pump();
      expect(tester.getTopLeft(boxFinder('c')), const Offset(0, 100));

      // x=25 is 0.25 of a cell: clear of the 0.5 + 0.2 deadband, so 'd' takes
      // the left slot and pushes 'c' across.
      await gesture.moveTo(const Offset(75, 150));
      await tester.pump();
      expect(tester.getTopLeft(boxFinder('c')), const Offset(100, 100));

      await gesture.up();
      await tester.pumpAndSettle();
    });

    testWidgets('lands the dropped tile in its slot', (
      WidgetTester tester,
    ) async {
      await tester.pumpWidget(
        host(
          columns: 2,
          width: 200,
          children: <ReorderGridTile>[box('a'), box('b'), box('c'), box('d')],
        ),
      );

      final TestGesture gesture = await beginDrag(tester, 'd');
      await gesture.moveTo(const Offset(60, 60));
      await tester.pumpAndSettle();
      await gesture.up();
      await tester.pumpAndSettle();

      expect(tester.getTopLeft(boxFinder('d')), Offset.zero);
    });

    testWidgets('keeps the dragged tile mounted from lift to landing', (
      WidgetTester tester,
    ) async {
      // A remount restarts whatever state the child holds, which is what makes
      // async-gated content flash while a tile is picked up and put down. The
      // grid never unmounts the tile it is carrying: the only extra mount is
      // the copy Draggable puts in the overlay, which Flutter requires.
      final _MountLog log = _MountLog();

      await tester.pumpWidget(
        host(
          columns: 2,
          width: 200,
          children: <ReorderGridTile>[
            box('a'),
            box('b'),
            box('c'),
            ReorderGridTile.count(
              key: const ValueKey<String>('d'),
              child: _LifecycleProbe(log: log),
            ),
          ],
        ),
      );

      expect(log.mounts, 1);

      final TestGesture gesture = await beginDrag(tester, 'd');
      await gesture.moveTo(const Offset(50, 50));
      await tester.pumpAndSettle();
      await gesture.up();
      await tester.pumpAndSettle();

      // One extra mount for the overlay copy, and that copy is the only thing
      // disposed. The tile's own state was never torn down.
      expect(log.mounts, 2);
      expect(log.disposals, 1);
    });

    testWidgets('shows the drop placeholder while a tile is carried', (
      WidgetTester tester,
    ) async {
      await tester.pumpWidget(
        host(
          columns: 2,
          width: 200,
          children: <ReorderGridTile>[box('a'), box('b'), box('c'), box('d')],
        ),
      );

      final Finder placeholder = find.descendant(
        of: find.byType(ReorderGrid),
        matching: find.byType(DecoratedBox),
      );
      expect(placeholder, findsNothing);

      final TestGesture gesture = await beginDrag(tester, 'd');
      await tester.pumpAndSettle();

      expect(placeholder, findsOneWidget);

      await gesture.up();
      await tester.pumpAndSettle();

      expect(placeholder, findsNothing);
    });

    testWidgets('ignores drags when reordering is disabled', (
      WidgetTester tester,
    ) async {
      int reorderCalls = 0;

      await tester.pumpWidget(
        host(
          columns: 2,
          width: 200,
          enableReorder: false,
          onReorder: (int from, int to) => reorderCalls++,
          children: <ReorderGridTile>[box('a'), box('b'), box('c'), box('d')],
        ),
      );

      expect(
        find.byWidgetPredicate(
          (Widget widget) =>
              widget.runtimeType.toString().startsWith('LongPressDraggable'),
        ),
        findsNothing,
      );

      final TestGesture gesture = await tester.startGesture(
        tester.getCenter(boxFinder('d')),
      );
      await tester.pump(_pastLongPress);
      await gesture.moveTo(const Offset(50, 50));
      await tester.pump(_pastPreview);
      await gesture.up();
      await tester.pumpAndSettle();

      expect(reorderCalls, 0);
      expect(tester.getTopLeft(boxFinder('d')), const Offset(100, 100));
    });
  });
}
