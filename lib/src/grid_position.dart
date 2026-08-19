/// A cell coordinate inside the grid.
///
/// Rows grow downwards along the main axis, columns grow along the cross axis.
/// Both are zero based.
typedef GridPosition = ({int row, int col});

/// The origin cell, used as a fallback when a tile has no placement yet.
const GridPosition kGridOrigin = (row: 0, col: 0);
