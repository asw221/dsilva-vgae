
library (dplyr)


.proj.dir <- file.path(.Dropbox(), "cancer", "dsilva")

.data.dir2 <- file.path( .proj.dir, "data" )
.out.dir <- file.path( "~", "standalone", "dsilva-vgae", "data", "raw" )


source (file.path(.proj.dir, "load-data01.R"))
ids.all <- sort(unique( dat$patient_ID ))



dat <- dplyr::select(
  dat,
  patient_ID,
  Cell_X_Position,
  Cell_Y_Position,
  Nucleus_Area,
  Nucleus_Axis_Ratio,
  Cytoplasm_Area__square_microns_,
  Membrane_ECAD__Cy3__Total__Norma,
  Cell_Area,
  Cell_Axis_Ratio,
  Cell_ECAD_Total,
  Cell_VIM_Total,
  Membrane_Cell_ECAD_Ratio,
  Nucleus_Cytoplasm_Area_Ratio
) %>%
  rename(
    x = "Cell_X_Position",
    y = "Cell_Y_Position",
    ID = "patient_ID"
  )



for ( id in ids.all ) {
  di <- dplyr::filter( dat, ID == id )
  write.csv(
    di, file = file.path(.out.dir, sprintf("cells_pat%d.csv", id)),
    row.names = FALSE
  )
}






