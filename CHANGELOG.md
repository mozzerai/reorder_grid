# Changelog

## 0.3.0

Transicoes de reordenacao mais previsiveis e continuas.

### Breaking

- `previewDelay` foi removido e substituido por `dragHysteresis` (double, em
  fracao de celula, padrao `0.2`). Nenhum consumidor passava `previewDelay`.

### Mudancas

- **Alvo por arredondamento.** O slot era escolhido com `floor` da quina
  superior esquerda do tile, entao era preciso arrastar uma celula inteira
  para o alvo mudar. Agora encaixa no slot mais proximo — troca na metade.
- **Preview sem espera.** O debounce de 150ms sumiu; o preview reage no mesmo
  frame. A protecao contra piscar na fronteira virou espacial
  (`dragHysteresis`) em vez de temporal, o que responde na hora e nao depende
  da velocidade do dedo. Some junto o `Timer` e o estado `_pendingAnchor`.
- **Voo de aterrissagem.** O tile solto desliza da posicao de soltura ate o
  slot, drenando elevacao e escala no caminho, em vez de sumir do overlay e
  reaparecer no lugar.
- **Decolagem animada.** O tile arrastado sobe para escala/elevacao em 140ms
  em vez de ja aparecer levantado.
- **Curvas.** Padroes passam de `300ms`/`easeInOut` para `220ms`/
  `easeOutCubic`. `easeInOut` comeca devagar, o que em movimento de objeto le
  como atraso.

### Interno

- `GridGeometry.snapAnchor` concentra o arredondamento e a zona morta, em Dart
  puro e coberto por testes.
- 10 testes novos (60 no total).

## 0.2.1

Sem mudanca de API.

- Remove todo `setState` do package. O que dispara repaint (ordem + posicoes)
  virou um `_GridSnapshot` imutavel atras de um `ChangeNotifier`, e so a
  `Stack` de tiles escuta, via `ListenableBuilder`. Um preview de arraste
  deixa de reconstruir o `LayoutBuilder` e o `DragTarget`.
- `_draggingKey` nao era lido no build — o `setState` no inicio do arraste era
  um rebuild a toa. Virou atribuicao simples.
- Atualizacoes vindas de `initState`/`didUpdateWidget` usam `setQuietly`, que
  nao notifica, porque o grid inteiro ja vai ser reconstruido no mesmo frame.

## 0.2.0

Reescrita interna do grid. A API pública continua compatível: `ReorderGrid.count`
e `ReorderGridTile.count` mantêm todos os parâmetros anteriores.

### Correções

- **Grid não reagia a mudança de largura.** As posições eram guardadas em pixels
  e só eram recalculadas quando `crossAxisCount` mudava, então redimensionar a
  janela (ou mudar `mainAxisSpacing`/`crossAxisSpacing`) deixava os tiles com a
  geometria antiga. Agora as posições são células e o pixel é derivado a cada
  build.
- **`childWhenDragging` nunca aparecia.** O tile arrastado era envolvido por um
  `AnimatedOpacity(opacity: 0)`, que também escondia o placeholder do slot de
  destino.
- **Animações trocavam de tile ao reordenar.** Os `AnimatedPositioned` não tinham
  key, então o estado da animação seguia o índice na `Stack` em vez do tile.
- **Tile mais largo que o grid apagava a tela inteira.** O empacotamento
  retornava `null` e nenhum tile recebia posição; agora o span é estreitado.
- **Reflow a cada rebuild do pai.** `ReorderGridTile.==` comparava o `child`, que
  quase nunca é igual entre builds, disparando um relayout completo em qualquer
  rebuild de ancestral. A comparação agora é estrutural (key + spans).
- **Dois grids na mesma tela trocavam tiles.** O payload do arraste era um `Key`
  cru, aceito por qualquer `DragTarget<Key>`; agora carrega a identidade do grid
  de origem.
- **Keys duplicadas faziam tiles sumirem em silêncio.** Passaram a lançar
  `FlutterError` em debug.
- Corrida entre `onDragEnd` e `onAcceptWithDetails` resolvida via
  `DraggableDetails.wasAccepted`, eliminando o flag `_dropHandled` e dois
  `addPostFrameCallback` com `setState`.

### Novidades

- `showSlotBorders` foi implementado (era um parâmetro morto) e agora tem
  padrão `false`.
- `ReorderGridTile.borderRadius` foi implementado (era um campo morto) e
  sobrescreve o raio do grid; o tipo passou a ser `double?`.
- `cellAspectRatio` para células não quadradas.
- `enableHapticFeedback` para desligar o retorno tátil.
- `animationCurve` e `previewDelay` configuráveis.
- `slotBorderColor` para a cor do contorno das células vazias.
- `ReorderGridCallback` exportado.

### Interno

- Arquivo único de 692 linhas dividido em `lib/src/`: `occupancy_grid.dart`,
  `dense_layout.dart`, `grid_geometry.dart`, `reorder_grid_tile.dart` e
  `reorder_grid.dart`. Empacotamento e geometria são Dart puro.
- 50 testes adicionados (antes: nenhum).
- Busca de slot livre reescrita sem `sync*` e com salto de linhas saturadas.
- Lints estritos (`strict-casts`, `strict-inference`, `strict-raw-types`,
  `public_member_api_docs`).

## 0.0.1

- Versão inicial.
