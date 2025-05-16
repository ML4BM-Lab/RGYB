import os
#os.environ['R_HOME'] = r'C:\Program Files\R\R-4.4.3'
import pandas as pd
import rpy2.robjects as ro
from rpy2.robjects import r, pandas2ri
from rpy2.robjects.packages import importr
from flask import Flask, render_template, request, render_template_string
from flask_bootstrap import Bootstrap5
from werkzeug.utils import secure_filename


app = Flask(__name__)
bootstrap = Bootstrap5(app)

@app.route("/", methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        f = request.files.get('archivo')
        try:
            if f:
                df = pd.read_csv(f)
                n= df.shape[1]
                if n ==16:
                    columnas_requeridas = {
                        "Age", "Roux limb", "Weight", "BMI", "Fat mass", "Waist circumference","Hip circumference", "REE (r)", "REE (t)", "Basal glycemia", "120' glycemia","HOMA", "Triglycerides", "Leptin", "TSH", "AP"
                    } 
                else:
                    columnas_requeridas = {
                                            "Age","Roux limb","Treitz","Ideal weight","Height","Weight","BMI","Fat percentage","Fat mass","Fat free mass percentage","Fat free mass","Waist circumference","Hip circumference","WHtR","Neck circumference","RQ","REE (r)","REE (t)","REE (t%)","Basal glycemia","30' glycemia","60' glycemia","90' glycemia","120' glycemia","Basal insulin","HOMA","30' insulin","60' insulin","90' insulin","120' insulin","Triglycerides","Cholesterol","HDL","LDL","Uric acid","CRP","Creatinine","Homocysteine","Leptin","TSH","Fibrinogen","Total bilirubin","Direct bilirubin","Indirect bilirubin","AST","ALT","AST/ALT ratio","AP","GGT","Erythrocytes","Haemoglobin","Haematocrit","Leucocytes","Neutrophils","Lymphocytes","Monocytes","Eosinophils","Basophils"
                        
                    }
                columnas_faltantes = columnas_requeridas - set(df.columns)
                if not columnas_faltantes:
                    file_path = "./models/mean_std2.csv"
                    dft = pd.read_csv(file_path)
                    #print(dft["variable"])
                    with(ro.default_converter + pandas2ri.converter).context():
                        dftc = ro.conversion.get_conversion().py2rpy(dft)
                        r.assign('test_data',dftc)
                    #r('print(test_data)')
                    with(ro.default_converter + pandas2ri.converter).context():
                        dfr = ro.conversion.get_conversion().py2rpy(df)
                        r.assign('dfr',dfr)
                    #r('print(dfr)') 
                    prediccionesx=0
                    if n== 16:
                        modelo_gb_bmi()
                        prediccionesx+=1
                        r('print("pasoooooooooo")')
                        modelo_rf_bmi()
                        prediccionesx+=1
                    else:
                        #modelo_gb_bmi()
                        #modelo_rf_bmi()
                        model_gb_ewl()
                        prediccionesx=1  

                    return f"Predicciones realizadas exitosamente: {prediccionesx}"
                else:
                    return f"Faltan las siguientes columnas: {', '.join(columnas_faltantes)}"
        except Exception as e:
            return f"Ocurrió un error al procesar el archivo: {str(e)}"
        
    return render_template('index.html')


def modelo_gb_bmi():
    r('''
        library('base')
        library('caret') 
        library('doParallel')
        library('gbm')
        library('pROC')
        library('renv')
        library('tidymodels')
        library('parallel')
        library('xgboost')
        library ('tidyr')
       
        
        #print(test_data)
      
        name_dfr = names(dfr)
        name_test = names(test_data)
        common <- intersect(name_test,name_dfr)
        ncommon <- length(common)
        #print(ncommon)
        #print(common)
      

        indices_common <- match(common,name_test) 
        means <- as.numeric(test_data[1,indices_common])
        sds <- as.numeric(test_data[2,indices_common])
        #print(means)
        #print(sds)
        #print(dfr)

        filas_con_na  <- sapply(dfr,function(x) which(is.na(x)))
        print(filas_con_na)
        dfp <- drop_na(dfr)
      
        dfs<-scale(dfr,center = means, scale = sds)
        dfs <- as.data.frame(dfs)
        names(dfs) <- names(dfr)[1:16]
        print("data centralizada")
        print(dfs)
        
      
        gb_bmi_toppred <-readRDS("./models/gb_bmi_toppred.RDS")
        df_class<-class(gb_bmi_toppred)
        print(df_class)

        print("modelo_gb_bmi")      
        prediccion <- gb_bmi_toppred %>%
        predict(new_data=dfs,type="prob")
        print(prediccion)
      
        #dfp$prediccion <- ifelse(prediccion$.pred_0 > prediccion$.pred_1,0,1)
        #dfp$significado <- ifelse(prediccion$.pred_0 < prediccion$.pred_1,"Fracaso en la cirugia","exito en la cirugia")

        dfp$Computed_probability_of_success <- prediccion$.pred_1
        dfp$Final_predicted_label <- ifelse(prediccion$.pred_1 > 0.4,1,0)
        dfp$Final_writen_label <- ifelse(prediccion$.pred_1 > 0.4,"success","failure")
        #print(dfp)
      
        nombre_archivo <- "/app/obesidad/gb_bmi_toppred_predict.csv"
        write.csv(dfp, file = nombre_archivo, row.names = FALSE)

      ''')
    
def modelo_rf_bmi():
    r('''
        library('caret') 
        library('doParallel')
        library('gbm')
        library('pROC')
        library('renv')
        library('tidymodels')
        library('parallel')
        library('xgboost')
        library('ranger')
        library ('tidyr')
      
        name_dfr = names(dfr)
        name_test = names(test_data)
        common <- intersect(name_test,name_dfr)
        ncommon <- length(common)
        #print(ncommon)
        #print(common)
      

        indices_common <- match(common,name_test)

        #print(test_data)
        means <- (test_data[1,indices_common])
        sds <- (test_data[2,indices_common])
      
        filas_con_na  <- sapply(dfr,function(x) which(is.na(x)))
        print(filas_con_na)
        dfp <- drop_na(dfr)
      
        dfs<-scale(dfp,center=means,scale=sds)
        dfs <- as.data.frame(dfs)
        names(dfs) <- names(dfr)[1:16]
        #print(dfs)
        
        rf_bmi_toppre <-readRDS("./models/rf_bmi_toppred.RDS")
        class(rf_bmi_toppre)
      
        print("rf_bmi_toppred")
        prediccion <- rf_bmi_toppre %>%
        predict(new_data=dfs,type="prob")
      
        print(prediccion)
      
        #dfp$prediccion <- ifelse(prediccion$.pred_0 > prediccion$.pred_1,0,1)
        #dfp$significado <- ifelse(prediccion$.pred_0 > prediccion$.pred_1,"Fracaso en la cirugia","exito en la cirugia")
        
        dfp$Computed_probability_of_success <- prediccion$.pred_1
        dfp$Final_predicted_label <- ifelse(prediccion$.pred_1 > 0.32,1,0)
        dfp$Final_writen_label <- ifelse(prediccion$.pred_1 > 0.32,"success","failure")
        print(dfp)
        
        nombre_archivo <- "/app/obesidad/rf_bmi_toppred_predict.csv"
        write.csv(dfp, file = nombre_archivo, row.names = FALSE)
      
      ''')

def model_gb_ewl():
    r('''
        library('caret') 
        library('doParallel')
        library('gbm')
        library('pROC')
        library('renv')
        library('tidymodels')
        library('parallel')
        library('xgboost')
        library('ranger')
        library ('tidyr')
      
    
        means <- (test_data[1,])
        sds <- (test_data[2,])
      
        filas_con_na  <- sapply(dfr,function(x) which(is.na(x)))
        print(filas_con_na)
        dfp <- drop_na(dfr)
      
        dfs<-scale(dfp,center=means,scale=sds)
        dfs <- as.data.frame(dfs)
        #print(dfs)
    
        #gb_ewl_allpred <- readRDS("./models/gb_ewl_allpred.RDS")
        gb_ewl_allpred <- readRDS("./models/d1.RDS")
        #class(gb_ewl_allpred)
      

        model_features <- gb_ewl_allpred$fit$feature_names 
        #print(model_features)          
        new_data_features <- colnames(dfs)
        #print(new_data_features)
        missing_features <- setdiff(model_features, new_data_features)
        print(missing_features)
      
        mismo_orden <- identical(model_features, new_data_features)

        if (mismo_orden) {
            print("Las columnas de ambos data frames tienen el mismo orden.")
        } else {
            print("Las columnas de los data frames NO tienen el mismo orden.")
        }
      
        print("gb_ewl_allpred")
      
        prediccion <- gb_ewl_allpred %>%
        predict(new_data=dfs,type="prob")

        #dfp$prediccion <- prediccion

        print(prediccion)
        
        #dfp$prediccion <- ifelse(prediccion$.pred_0 > prediccion$.pred_1,0,1)
        #dfp$significado <- ifelse(prediccion$.pred_0 > prediccion$.pred_1,"Fracaso en la cirugia","exito en la cirugia")

        dfp$Computed_probability_of_success <- prediccion$.pred_1
        dfp$Final_predicted_label <- ifelse(prediccion$.pred_1 > 0.4,1,0)
        dfp$Final_writen_label <- ifelse(prediccion$.pred_1 > 0.4,"success","failure")
        #print(dfp)
      
        nombre_archivo <- "/app/obesidad/gb_ewl_allpred_predict.csv"
        write.csv(dfp, file = nombre_archivo, row.names = FALSE)
    ''')

if __name__ == '__main__':
    app.run(debug=True, port=5002)
