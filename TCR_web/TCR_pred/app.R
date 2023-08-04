library(shiny)
library(tidyverse)
library(DT)
library(shinyjs)
library(shinythemes)
library(shinydashboard)
library(dashboardthemes)
library(shinycssloaders)

options(spinner.color = "#0dc5c1",
        spinner.type = 4)

################################# Change ab_path ####################################
# 请提供到TCR_web的绝对路径 - Please provide the absolute path to TCR_web dir
ab_path <- "~/R/TCR_Researchs/TCR_web/"
#####################################################################################

# 读取文件
cd4_data <- read_csv(paste0(ab_path, "data/cd4_data.csv"))
cd4_cdr3 <- cd4_data$CDR3
cd4_pep <- cd4_data$Epitope

#####################################################################################
HomePage <- dashboardPage(
              dashboardHeader(disable = T),
              dashboardSidebar(
                tags$head(tags$style(HTML(".sidebar { font-size: 15px; /* 调整文字大小 */}"))),
                sidebarMenu(
                menuItem("Single mode", tabName = "single_mode"),
                menuItem("Batch mode", tabName = "batch_mode")
              )),
              dashboardBody(
                            h2("Deep learning-based prediction of CD4 T-cell receptor specificity",
                               align = "center"),
                            br(),
                            tabItems(
                              tabItem(tabName = "single_mode",
                                      column(
                                        width = 12,  # 设置列的宽度，根据需要进行调整
                                        offset = 1,  # 设置列的偏移量，以实现居中对齐
                                        box(
                                          width = 10,
                                          h3("Single Mode"),
                                          br(),
                                          h4("Please input sequences: "),
                                          textInput("cdr3", label = "CDR3", placeholder = "CDR3", width = '50%'),
                                          textInput("peptide", label = "Peptide", placeholder = "Peptide", width = '50%'),
                                          actionButton("single_pre", "RUN"),
                                          helpText("This is single mode, the run result is as below"),
                                          strong(textOutput("single_res") %>% withSpinner(proxy.height = "20vh", size = 0.7,
                                                                                          caption = div(strong("Loading"), br(), em("Please wait"))))
                                        ))),
                              tabItem(tabName = "batch_mode",
                                      column(
                                        width = 12,  # 设置列的宽度，根据需要进行调整
                                        offset = 1,  # 设置列的偏移量，以实现居中对齐
                                        box(
                                          width = 10,
                                          h3("Batch Mode"),
                                          br(),
                                          h4("Please select and upload a file"),
                                          fileInput("upload_file", label = NULL, width = '50%'),
                                          column(width = 10,
                                            fluidRow(actionButton("batch_pre", "RUN"),
                                                     downloadButton("download", "Download"),
                                                     textOutput("batch_warning"))),
                                          column(width = 10,
                                                 fluidRow(helpText("If you do not know the format of input file, please refer Help page"))),
                                          column(width = 12,
                                                 fluidRow(DT::dataTableOutput("batch_res") %>% withSpinner(proxy.height = "200px",
                                                                                                           caption = div(strong("Loading"), br(), em("Please wait")))))
                                        )))),
                            div(
                              style = "position: fixed; bottom: 0; left: 55%",
                              HTML("<p>© Liuxs-Lab 2023</p>")
                            )))

SearchPage <- fluidPage(
  fluidRow(box(width = 4,
               h4("Search parameters"),
               strong("Sequence type: "),
               selectInput("seqType", label = NULL, choices = c("CDR3" = "cdr3", "Epitope" = "pep"), selected = "cdr3", width = "50%"),
               strong("Input sequence below: "),
               textInput("seq", NULL, value = "", width = "50%", placeholder = "Input sequence"),
               strong("The maximum Levenshtein distance(Levenshtein distance refers to the edit distance between two sequences): "),
               numericInput("number", NULL, value = 0, min = 0, max = 10, step = 1, width = "50%"),
               column(width = 12,
                      fluidRow(actionButton("search", "Search"),
                               HTML(paste0(rep("&nbsp;", 10), collapse = "")),
                               actionButton("clear", "Clear"))),
               p("Tip: When initiating a search without specifying a sequence, the system retrieves and presents the complete contents of the database."),
               br(),
               strong("Note: Those are collected positive CD4 TCRB CDR3 data, column of Score is obtained from our model.")
               ),
           box(width = 8, 
               DT::dataTableOutput("table") %>% withSpinner(proxy.height = "400px",
                                                            caption = div(strong("Loading"), br(), em("Please wait")))),
           div(
             style = "position: fixed; bottom: 0; left: 50%",
             HTML("<p>© Liuxs-Lab 2023</p>")
           )
))

#####################################################################################
ui <- navbarPage(title = "CD4 TCR specifity prediction",
                 theme = "flatly",
                 tabPanel("Home", HomePage),
                 tabPanel("Search", SearchPage),
                 tabPanel("Help", ""),
                 tabPanel("About", ""))

server <- function(input, output, session) {
  global_val <- reactiveValues(
    single_score = NULL,
    single_warning = NULL,
    batch_int = NULL,
    batch_warning = NULL,
    seqType = NULL,
    seq = NULL,
    number = NULL,
    cal_df = cd4_data  # 计算得到的数据表--编辑距离, 初始是cd4_data(全部数据)
  )
  
  single.reactiveValues.reset <- function() {
    global_val$single_score  <-  NULL
    global_val$single_warning  <-  NULL
  }
  
  batch.reactiveValues.reset <- function() {
    global_val$batch_int  <-  NULL
    global_val$batch_warning  <-  NULL
  }
  
  distance.reactiveValues.reset <- function() {
    global_val$seqType  <-  NULL
    global_val$seq  <-  NULL
    global_val$number <- NULL
    global_val$cal_df <- NULL
  }
  
  single_cal <- reactive({
    script <- paste0(ab_path, "TCR_script_web/tcr_pre.py")
    cmd <- paste("python", script, "--mode single --cdr3", input$cdr3,
                 "--pep", input$peptide)
    system(paste0(". ~/miniconda3/bin/activate;conda activate machine_learning_torch;", cmd),
           intern = T)
  })
  
  observeEvent(input$single_pre, {
    single.reactiveValues.reset() # 重置
    if (!(input$peptide=="" | input$cdr3=="")) {
      global_val$single_score <- single_cal()
    } else {
      global_val$single_warning <- "Please input CDR3 and peptide!!!"
    }
  })
  
  batch_cal <- reactive({
    data_path <- input$upload_file$datapath # 获取上传文件路径
    script <- paste0(ab_path, "TCR_script_web/tcr_pre.py")
    cmd <- paste("python", script, "--mode batch --data_path", data_path)
    system(paste0(". ~/miniconda3/bin/activate;conda activate machine_learning_torch;", cmd),
           intern = T)
  })
  
  observeEvent(input$batch_pre, {
    batch.reactiveValues.reset()
    if (!is.null(input$upload_file)) { 
      global_val$batch_int <- batch_cal()
    } else {
      global_val$batch_warning <- "Please upload a file that meet required format!!!"
    }
  })
  
  dist_cal <- reactive({
    tmp <- switch(input$seqType,
                  cdr3 = cd4_cdr3,
                  pep = cd4_pep)
    dist_res <- adist(input$seq, tmp, counts = TRUE)
    sum_dist <- as.data.frame(as.numeric(dist_res))
    count_dist <- as.data.frame(drop(attr(dist_res, "counts")))
    mask <- sum_dist <= input$number
    df <- cbind.data.frame(sum_dist, count_dist, cd4_data)
    colnames(df)[1] <- "Edit distance"
    df[mask, ]
  })
  
  observeEvent(input$search, {
    distance.reactiveValues.reset()
    if (input$seq == "") {
      global_val$seqType <- input$seqType
      global_val$number <- input$number
      global_val$cal_df <- cd4_data
    } else {
      global_val$seqType <- input$seqType
      global_val$seq <- input$seq
      global_val$number <- input$number
      global_val$cal_df <- dist_cal()
    }
  })
  
  observeEvent(input$clear, {
    distance.reactiveValues.reset()
    updateTextInput(session, "seq", value = "")
    updateSelectInput(session, "seqType", selected = "cdr3")
    updateNumericInput(session, "number", value = 0)
    global_val$cal_df <- cd4_data
  })

##################################################################################### 
  output$single_res <- renderText(
    if (!is.null(global_val$single_score)) {
      paste("Prediction score is ", global_val$single_score)
  } else {
    global_val$single_warning
  })
  
  output$batch_res <- renderDataTable(
    if (!is.null(global_val$batch_int)) {
      batch_path <- paste0(ab_path, "tmp/batch_",
                           global_val$batch_int, ".csv")
      read_csv(batch_path)
    }, rownames = F,
    options = list(
      pageLength = 10,
      lengthMenu = list(c(10, 50, 100,-1), c('10', '50','100', 'All')),
      scrollY = "18vh",
      scrollX = "100%",
      scrollCollapse = FALSE,
      columnDefs = list(
        list(
          targets = "_all",
          className = "dt-left"  # 所有列的内容左对齐
        ))))
  
  output$batch_warning <- renderText({
    if (is.null(global_val$batch_int)) {
      global_val$batch_warning
    }
  })
  
  output$download <- downloadHandler( # 下载文件
    filename = paste0("Pep2TCR-", Sys.Date(), ".csv"),
    content = function(file) {
      if (!is.null(global_val$batch_int)) {
        batch_path <- paste0(ab_path, "tmp/batch_",
                             global_val$batch_int, ".csv")
        write_csv(read_csv(batch_path), file)
      } else {
        NULL  # 报错
      }
    }
  )
  
  output$table <- renderDataTable(
    { global_val$cal_df },
    rownames = F, 
    extensions = 'Buttons',
    options = list(
      pageLength = 50,
      lengthMenu = list(c(10, 50, 100, -1), c('10', '50','100', 'All')),
      scrollY = "64vh",
      scrollX = "100%",
      scrollCollapse = FALSE,
      columnDefs = list(
        list(
          targets = "_all",
          className = "dt-left"  # 所有列的内容左对齐
        )),
      dom = '<"row"<"col-sm-3"B><"col-sm-3"l><"col-sm-6"f>>rtip',
      buttons = list(
        list(
          extend = 'csv',
          text = 'Download current page',
          className = 'btn-primary'
        ))))
}

# Run the application 
shinyApp(ui = ui, server = server)
