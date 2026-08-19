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

Widget host({
  required List<ReorderGridTile> children,
  double width = 400,
  int columns = 4,
  double spacing = 0,
  bool enableReorder = true,
  bool showSlotBorders = false,
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
            onReorder: onReorder,
            children: children,
          ),
        ),
      ),
    ),
  );
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
