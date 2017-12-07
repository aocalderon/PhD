#!/usr/bin/Rscript

args = commandArgs(trailingOnly = TRUE)

rmarkdown::render(
  "plotScaleupFromLogs2.Rmd",
  "all",
  output_format="pdf_document", 
  output_file = paste0(args[2],".pdf"),
  params = list(
    phd_path = args[1],
    filename = args[2],
    extension = args[3]
  )  
)
rmarkdown::render(
  "plotScaleupFromLogs2.Rmd",
  "all",
  output_format="html_document", 
  output_file = paste0(args[2],".html"),
  params = list(
    phd_path = args[1],
    filename = args[2],
    extension = args[3]
  )  
)
