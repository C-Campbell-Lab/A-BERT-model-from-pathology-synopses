library(circlize)
library(rlang)
set.seed(42)

tmp = read.csv(file = 'mat_abbr.csv', row.names = 1)

mat <- as.matrix(tmp)

colnames(mat) = rownames(mat)

grid_col = rand_color(21, transparency = 0.5)

names(grid_col) = rownames(mat)

pdf(file = "e://chord.pdf", width=10, height=10)
par(cex = 1, mar = c(0, 0, 0, 0))
chordDiagram(t(mat), annotationTrack = "grid",symmetric = TRUE, grid.col=grid_col, row.col=grid_col,
    preAllocateTracks = list(track.height = max(strwidth(unlist(dimnames(mat))))))
circos.track(track.index = 1, panel.fun = function(x, y) {
    circos.text(CELL_META$xcenter, CELL_META$ylim[1], CELL_META$sector.index,
        facing = "clockwise", niceFacing = TRUE, adj = c(0, 0.5))
}, bg.border = NA)
dev.off()

pdf(file = "e://chord_mds.pdf", width=10, height=10)
par(cex = 1, mar = c(0, 0, 0, 0))
grid_col_mds = duplicate(grid_col, shallow = FALSE)
for (idx in 0:21) {
  if(idx==18) next
   grid_col_mds[idx] =  "#00000000"
}
chordDiagram(mat, annotationTrack = "grid",symmetric = TRUE, grid.col=grid_col,  row.col=grid_col_mds,
    preAllocateTracks = list(track.height = max(strwidth(unlist(dimnames(mat))))))
circos.track(track.index = 1, panel.fun = function(x, y) {
    circos.text(CELL_META$xcenter, CELL_META$ylim[1], CELL_META$sector.index,
        facing = "clockwise", niceFacing = TRUE, adj = c(0, 0.5))
}, bg.border = NA)
dev.off()

pdf(file = "e://chord_mpn.pdf", width=10, height=10)
par(cex = 1, mar = c(0, 0, 0, 0))
grid_col_mpn = duplicate(grid_col, shallow = FALSE)
for (idx in 0:21) {
  if(idx==19) next
   grid_col_mpn[idx] =  "#00000000"
}
chordDiagram(mat, annotationTrack = "grid",symmetric = TRUE, grid.col=grid_col,  row.col=grid_col_mpn,
    preAllocateTracks = list(track.height = max(strwidth(unlist(dimnames(mat))))))
circos.track(track.index = 1, panel.fun = function(x, y) {
    circos.text(CELL_META$xcenter, CELL_META$ylim[1], CELL_META$sector.index,
        facing = "clockwise", niceFacing = TRUE, adj = c(0, 0.5))
}, bg.border = NA)
dev.off()

pdf(file = "e://chord_nor.pdf", width=10, height=10)
par(cex = 1, mar = c(0, 0, 0, 0))
grid_col_nor = duplicate(grid_col, shallow = FALSE)
for (idx in 0:21) {
  if(idx==20) next
   grid_col_nor[idx] =  "#00000000"
}
chordDiagram(mat, annotationTrack = "grid",symmetric = TRUE, grid.col=grid_col,  row.col=grid_col_nor,
    preAllocateTracks = list(track.height = max(strwidth(unlist(dimnames(mat))))))
circos.track(track.index = 1, panel.fun = function(x, y) {
    circos.text(CELL_META$xcenter, CELL_META$ylim[1], CELL_META$sector.index,
        facing = "clockwise", niceFacing = TRUE, adj = c(0, 0.5))
}, bg.border = NA)
dev.off()

# library(htmlwidgets)
# library(chorddiag)
# # Create the chord diagram.
# w <-  chorddiag(mat,  type = "directional", showTicks = F, groupnameFontsize = 14, groupnamePadding = 10, margin = 90)
# saveWidget(
#   w,
#   'e://chord.html',
#   selfcontained = TRUE,
#   libdir = NULL,
#   background = "white",
# )
