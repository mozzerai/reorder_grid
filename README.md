# reorder_grid

Grid denso e reordenável para Flutter. Empacota tiles de tamanhos variados sem
deixar buracos, anima a reacomodação e mostra um preview ao vivo enquanto o
usuário arrasta.

## Uso

```dart
ReorderGrid.count(
  crossAxisCount: 4,
  mainAxisSpacing: 8,
  crossAxisSpacing: 8,
  onReorder: viewModel.reorder, // (oldIndex, newIndex)
  children: [
    for (final item in items)
      ReorderGridTile.count(
        key: ValueKey(item.id),
        crossAxisCellCount: item.width,  // colunas
        mainAxisCellCount: item.height,  // linhas
        child: ItemCard(item),
      ),
  ],
)
```

O grid se dimensiona pela altura do conteúdo e exige largura limitada — a ideia
é colocá-lo dentro de um scrollable (`SliverToBoxAdapter`,
`SingleChildScrollView`), não fazer ele rolar sozinho.

## Como o layout funciona

As posições são guardadas em **células**, não em pixels. Cada build converte
célula → pixel a partir das constraints recebidas, então mudar largura, spacing
ou aspect ratio reacomoda o grid sozinho, sem estado a invalidar.

O empacotamento é *first-fit* na ordem da lista: cada tile ocupa a primeira
posição livre de cima para baixo, da esquerda para a direita. Isso mantém a
ordem visual igual à ordem da lista e ainda preenche os buracos deixados por
tiles altos.

Durante o arraste o tile em movimento é fixado sob o ponteiro e os demais são
reempacotados ao redor. Soltar confirma o preview e dispara `onReorder` com os
índices em ordem de leitura; soltar fora do grid restaura o arranjo anterior.

## Como o arraste responde

O tile **encaixa no slot mais próximo**: a troca acontece quando ele passa da
metade da célula, não quando cobre a célula inteira. Isso é o que faz o alvo
coincidir com a intuição de onde o tile "está".

Em cima disso vem `dragHysteresis` — uma zona morta (padrão: 0,2 célula) que o
tile precisa ultrapassar antes de largar o slot atual. Sem ela, parar o dedo em
cima da fronteira faria o preview piscar entre dois arranjos. Com ela, o preview
reage no mesmo frame, sem tempo de espera.

O tile arrastado **nunca é desmontado**. O grid não usa `childWhenDragging` —
trocar o filho desmontaria a subárvore, e remontar reiniciaria o estado que ela
carrega (um gate assíncrono volta para o ramo pendente, uma animação recomeça).
O conteúdo fica no lugar, invisível, atrás do placeholder do slot.

A única cópia extra é a que o `Draggable` do Flutter coloca no `Overlay` para
flutuar sob o dedo; essa é inevitável.

## Parâmetros

| Parâmetro | Padrão | Descrição |
|---|---|---|
| `crossAxisCount` | — | Número de colunas (máximo 62). |
| `children` | — | Tiles em ordem de leitura. Keys únicas. |
| `mainAxisSpacing` / `crossAxisSpacing` | `8.0` | Espaçamento entre linhas / colunas. |
| `cellAspectRatio` | `1.0` | Largura ÷ altura da célula, como em `GridView.childAspectRatio`. |
| `enableReorder` | `true` | Desliga toda a maquinaria de arraste. |
| `enableHapticFeedback` | `true` | Haptics no início do arraste e a cada mudança de preview. |
| `showSlotBorders` | `false` | Contorna as células vazias. |
| `slotBorderColor` | `colorScheme.outlineVariant` | Cor desse contorno. |
| `onReorder` | `null` | `(oldIndex, newIndex)` após um drop ou ação de acessibilidade que mudou a posição. |
| `borderRadius` | `8.0` | Raio padrão dos tiles (`ReorderGridTile.borderRadius` sobrescreve). |
| `animationDuration` / `animationCurve` | `220ms` / `easeOutCubic` | Reacomodação dos tiles. |
| `dragHysteresis` | `0.2` | Zona morta, em fração de célula, antes do preview trocar de slot. `0` troca na metade exata. |

## Rolagem durante o arraste

Dentro de um scrollable, o grid rola sozinho quando o dedo chega a 48 px da
borda da viewport. A velocidade cresce de zero até 900 px/s conforme o dedo se
aproxima da borda, e o conteúdo avança **uma vez por frame**, pelo tempo
decorrido — a mesma distância em todo frame. O `EdgeDraggingAutoScroller` do
Flutter encadeia uma animação curta por passo e deixa um frame parado a cada
três, o que aparece como tranco.

A zona segue o **ponteiro**, não o tile: um card do dashboard pode ser mais alto
que a tela, e um retângulo desse tamanho encostaria nas duas bordas ao mesmo
tempo.

**O preview fica congelado enquanto a borda rola.** Reempacotar a cada linha que
passa sob o dedo faria os outros cards pularem de um lado para o outro várias
vezes por segundo — o empacotamento denso não se move de forma monotônica
quando o tile fixado desce uma linha. O preview é recalculado uma vez, quando a
rolagem para (o dedo saiu da zona ou o conteúdo acabou), e soltar no meio da
rolagem mira a posição real do tile, não o preview congelado.

## Leitor de tela

Cada tile oferece ações de acessibilidade para mover para o início, uma posição
para trás, uma para frente e para o fim — os mesmos textos do
`ReorderableListView`, vindos de `WidgetsLocalizations`. A ação move o tile na
ordem da lista, reempacota o grid e dispara `onReorder` como um drop.

## Limitações conhecidas

- **Sem reordenação por teclado.** Leitor de tela usa as ações acima; teclado
  físico ainda não tem atalhos.
- **Máximo de 62 colunas**, imposto pelo bitmask de ocupação (uma linha = um
  `int`). Valores maiores são reduzidos.
- Tiles mais largos que o grid são **estreitados** até a largura total em vez de
  quebrarem o layout.

## Desenvolvimento

```bash
flutter pub get
flutter analyze
flutter test
dart format .
```

O algoritmo de empacotamento (`lib/src/occupancy_grid.dart`,
`lib/src/dense_layout.dart`) e a conversão célula → pixel
(`lib/src/grid_geometry.dart`) são Dart puro, sem dependência de widget, e
concentram a maior parte dos testes.
