rm(list = ls())

library(xtable)
library(reshape2)

# Change file name here %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
filename <- "../../results/tabular_processed/processed_results-SHAP-ExErr.csv"

df = read.csv(filename, header=TRUE, sep=',')
  
# Local or Global group (CHANGE HERE) %%%%%%%%%%%%%%%%%%%%%%%%%%%
df <- df[,grep(pattern="^Global|Dataset", colnames(df))]

# Depending on the ground truth, there is some explainers that are left out
if (length(grep(pattern="gradient", filename))==1){
    df <- df[,!grepl(pattern="^(Global|Local)\\.(SHAP\\.|PE_adj\\.)", colnames(df))]
} else {
    df <- df[,!grepl(pattern="^(Global|Local)\\.(PE\\.|SHAP_adj\\.|LIME\\.)", colnames(df))]
}

df_melted <- melt(df,
  id.vars       = "Dataset",
  variable.name = "regressorNexplainer",
  value.name    = "meanRMSE"
)
df_melted["regressorNexplainer"] <- lapply(df_melted["regressorNexplainer"], function(x) {gsub("Global.", "", gsub("Local.", "", x))})

res <- pairwise.wilcox.test( df_melted$meanRMSE, df_melted$regressorNexplainer, p.adjust.method='bonferroni') 

res$p.value.formatted = format(res$p.value,scientific=T,digits=2)
res$p.value.formatted = ifelse(res$p.value < 0.05, paste0(res$p.value.formatted, "*"),res$p.value.formatted)
res$p.value.formatted = ifelse(res$p.value < 0.01, paste0(res$p.value.formatted, "*"),res$p.value.formatted)
res$p.value.formatted = ifelse(res$p.value < 0.001, paste0(res$p.value.formatted, "*"),res$p.value.formatted)

ltx <- xtable(res$p.value.formatted, caption="",  caption.placement="top", sanitize.text.function = identity)

print(ltx, sanitize.text.function = identity, caption.placement="top")

