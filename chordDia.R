library(circlize)

tmp = read.csv(file = 'mat_abbr.csv', row.names = 1)

mat <- as.matrix(tmp)

colnames(mat) = rownames(mat)

pdf(file = "e://chord.pdf", width=10, height=10)
par(cex = 1, mar = c(0, 0, 0, 0))
chordDiagram(mat, annotationTrack = "grid",symmetric = TRUE,
    preAllocateTracks = list(track.height = max(strwidth(unlist(dimnames(mat))))))
circos.track(track.index = 1, panel.fun = function(x, y) {
    circos.text(CELL_META$xcenter, CELL_META$ylim[1], CELL_META$sector.index,
        facing = "clockwise", niceFacing = TRUE, adj = c(0, 0.5))
}, bg.border = NA)
dev.off()


library(htmlwidgets)
library(chorddiag)
# Create the chord diagram.
w <-  chorddiag(mat,  type = "directional", showTicks = F, groupnameFontsize = 14, groupnamePadding = 10, margin = 90)
saveWidget(
  w,
  'e://chord.html',
  selfcontained = TRUE,
  libdir = NULL,
  background = "white",
)
