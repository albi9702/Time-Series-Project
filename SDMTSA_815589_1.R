#-- Librerie ----
library("tidyverse")
library("xts")
library("forecast")
library("KFAS")
library("lubridate")
library("caret")
library("astsa")
library("urca")
library("tsfknn")
library("ModelMetrics")
library("keras")
library("DMwR")
library("ggpubr")
library("dygraphs") #-- Dynamic Graphs

#-- Lettura File ----
ts <- read_csv2("G:\\Il mio Drive\\Università\\Data Science\\2° Anno\\Streaming Data Management and Time Series Analysis\\Esame\\TrainingSet.csv")
ts <- ts %>%
  rename(Data = DATA,
         Valore = VALORE)
head(ts)

#-- Analisi Esplorativa ----
#-- 1. Problema Ora Legale ----
ts %>%
  group_by(year(date(Data)),
           month(date(Data)),
           day(date(Data)))          %>% #-- Raggruppamento
  summarise(n())                     %>% #-- Somma per Righe
  filter(`n()` == 23)                %>% #-- Filtro (Ora Legale)
  rename(Year = "year(date(Data))",
         Month = "month(date(Data))",
         Day = "day(date(Data))",
         `Day Hours` = "n()")            #-- Rinomina Colonne

ts_legale <- tribble(
  ~Data, ~Ora, ~Valore,
  "2019-03-31", 3, 3039997,
  "2020-03-29", 3, 2329514
) %>%
  mutate(Data = as.Date(Data))

ts <- ts %>%
  bind_rows(ts_legale) %>%
  arrange(Data)

#-- 2. Stagionalità ----

#-- Gionaliera
ts_daily <- ts                   %>% #-- Serie Storica
            bind_rows(ts_legale) %>% #-- Concatenazione con Ora Legale
            arrange(Data)        %>% #-- Ordinamento per Sistemare Ora Legale
            ts(frequency = 24)       #-- Oggetto `ts()` con Frequenza Giornaliera

ggseasonplot(ts_daily[, "Valore"],                      #-- Stagionalità Giornaliera
             year.labels = TRUE,                        #-- Etichetta
             year.labels.left = FALSE,                  #-- Etichetta
             ylab = "Valore (in milioni)",              #-- Etichetta Asse Y
             xlab = "Orario del Giorno",                #-- Etichetta Asse X
             main = "Grafico Stagionale Giornaliero") + #-- Titolo
  theme(axis.text.y = element_text(face = "bold"),      #-- Grassetto per Asse Y
        axis.text.x = element_text(face = "bold",       #-- Grassetto per Asse X
                                   angle = 45)) +       #-- 45°
  scale_y_continuous(breaks = c(2000000,
                                4000000,
                                6000000,
                                8000000),               #-- Interruzioni
                     labels = c(2,4,6,8)) +             #-- Etichetta
  theme_bw()                                            #-- Tema

#-- Settimanale
ts_weekly <- ts                   %>% #-- Serie Storica
             bind_rows(ts_legale) %>% #-- Concatenazione con Ora Legale
             arrange(Data)        %>% #-- Ordinamento per Sistemare Ora Legale
             ts(frequency = 24*7)     #-- Oggetto `ts()` con Frequenza Settimanale

ggseasonplot(ts_weekly[, "Valore"],                     #-- Stagionalità Settimanale
             year.labels = TRUE,                        #-- Etichetta
             year.labels.left = FALSE,                  #-- Etichetta
             ylab = "Valore (in milioni)",              #-- Etichetta Asse Y
             xlab = "Giorni in una Settimana",          #-- Etichetta Asse X
             main = "Grafico Stagionale Settimanale") + #-- Titolo
  theme(axis.text.y = element_text(face = "bold"),      #-- Grassetto per Asse Y
        axis.text.x = element_text(face = "bold",       #-- Grassetto per Asse X
                                   angle = 45)) +       #-- 45°
  scale_y_continuous(breaks = c(2000000,
                                4000000,
                                6000000,
                                8000000),               #-- Interruzioni
                     labels = c(2,4,6,8)) +             #-- Etichetta
  theme_bw()                                            #-- Tema

#-- Mensile
ag <- aggregate(Valore ~ Data, ts, mean)

ggseasonplot(ts(ag[,"Valore"], frequency = 30),         #-- Stagionalità Mensile
             year.labels = TRUE,                        #-- Etichetta
             year.labels.left = FALSE,                  #-- Etichetta
             ylab = "Valore (in milioni)",              #-- Etichetta Asse Y
             xlab = "Giorni in un Mese",                           #-- Etichetta Asse X
             main = "Grafico Stagionale Mensile") +     #-- Titolo
  theme(axis.text.y = element_text(face = "bold"),      #-- Grassetto per Asse Y
        axis.text.x = element_text(face = "bold",       #-- Grassetto per Asse X
                                   angle = 45))   +     #-- 45°
  scale_y_continuous(breaks = c(3000000,
                                4000000,
                                5000000,
                                6000000),               #-- Interruzioni
                     labels = c(3,4,5,6)) +             #-- Etichetta
  theme_bw()                                            #-- Tema

#-- Annuale
ggseasonplot(ts(ag[,"Valore"], frequency = 730/2),      #-- Stagionalità Annuale
             year.labels = TRUE,                        #-- Etichetta
             year.labels.left = FALSE,                  #-- Etichetta
             ylab = "Valore (in milioni)",              #-- Etichetta Asse Y
             xlab = "Giorni in un Anno",                           #-- Etichetta Asse X
             main = "Grafico Stagionale Annuale") +     #-- Titolo
  theme(axis.text.y = element_text(face = "bold"),      #-- Grassetto per Asse Y
        axis.text.x = element_text(face = "bold",       #-- Grassetto per Asse X
                                   angle = 45)) +       #-- 45°
  scale_y_continuous(breaks = c(3000000,
                                4000000,
                                5000000,
                                6000000),               #-- Interruzioni
                     labels = c(3,4,5,6)) +             #-- Etichetta
  theme_bw()                                            #-- Tema

#-- Scomposizione Train e Validation ----

ts_stand <- ts(scale(ts$Valore), #-- Standardizzazione
               frequency = 24,   #-- Frequenza
               start = 1)        #-- Parte da 2018-09-01

train_arima <- ts(ts_stand[1:as.integer(nrow(ts)*0.8), ], #-- Train
                  frequency = 24,                         #-- Frequenza
                  start = 1)                              #-- Parte da 2018-09-01

validation_arima <- ts(ts_stand[(1 + as.integer(nrow(ts)*0.8)) : nrow(ts), ], #-- Validation
                       frequency = 24,                                        #-- Frequenza
                       start = 585)                                           #-- Parte da 2020-03-01

#-- Dataset per Grafico Interattivo

#-- Grafico Interattivo
dyn_train <- dyn_ts[1:as.integer(nrow(dyn_ts)*0.8), ]/1000000
dyn_validation <- dyn_ts[(1 + as.integer(nrow(dyn_ts)*0.8)) : nrow(dyn_ts), ]/1000000

dyn_all <- cbind(dyn_train,
                 dyn_validation)
colnames(dyn_all) <- c("Train",
                       "Validation")

dygraph(dyn_all)                             %>%
  dySeries("Train",
           label = "Train")                  %>% #-- Train Dataset
  dySeries("Validation",
           label = "Validation")             %>% #-- Validation Dataset
  dyOptions(drawPoints = TRUE,                   #-- Draw Points
            pointSize = 0.5,                     #-- Size of Points
            colors = c("blue",
                       "red"))               %>% #-- Colori
  dyAxis("y", label = "Valore (in Milioni)") %>% #-- Asse Y
  dyAxis("x", label = "Periodo",
         drawGrid = FALSE)                   %>% #-- Asse X
  dyRangeSelector(height = 20)                   #-- Slider

#-- Grafico Previsione + Reale
plot_val <- function(prediction, title){
  
  autoplot(train_arima,
           main = title,                          #-- Titolo
           xlab = "Periodo",                      #-- Label X
           ylab = "Valore (in Milioni)",
           col = "#0077b6")        + #-- Label Y
    autolayer(validation_arima,                   
              series = "Reale")                 + #-- Validation
    autolayer(prediction,
              series = "Previsione")            + #-- Previsione
    scale_y_continuous(breaks = c(-2, 0, 2, 4),   
                       labels = c(2, 4, 6, 8))  + #-- Interruzioni
    scale_color_manual(values = c("#248232",
                                  "#c81d25"))   + #-- Colori
    labs(color = "Validation")                  + #-- Legenda
    theme_bw()                                    #-- Tema
}

#-- Naive Method ----
naive <- stlf(train_arima, h = 3504)

plot_val(naive$mean,
         "Previsioni Naive Method")

#-- ARIMA ----

#-- ACF e PACF ----
acf  <- ggAcf(ts[, "Valore"],
        lag.max = 72, 
        main = "")
pacf <- ggPacf(ts[, "Valore"],
        lag.max = 72,
        main = "")

ggarrange(acf, pacf,
          labels = c("ACF", "PACF"),
          nrow = 2, ncol = 1)

#-- Differenze Giornaliere
train_diff <- diff(train_arima,
                   lag = 24)        %>%
              diff(differences = 1) %>%
              diff(differences = 1)

acf_diff  <- ggAcf(train_diff,
                   lag.max = 72,
                   main = "")
pacf_diff <- ggPacf(train_diff,
                    lag.max = 72,
                    main = "")

ggarrange(acf_diff,
          pacf_diff,
          labels = c("ACF", "PACF"),
          nrow = 2, ncol = 1)

Box.test(train_diff,
         type='Ljung-Box')

#--- 1. ARIMA(2, 0, 0)(0, 1, 0) ----
mod1_arima <- Arima(train_arima,
                    c(2, 0, 0),
                    c(0, 1, 0),
                    lambda = "auto",
                    include.constant = TRUE)
smod1_arima <- summary(mod1_arima)
smod1_arima[,"MAE"]

#-- ACF e PACF
acf_arima1 <- ggAcf(mod1_arima$residuals,
                    lag.max = 96,
                    main = "")
pacf_arima1 <- ggPacf(mod1_arima$residuals,
                      lag.max = 96,
                      main = "")

ggarrange(acf_arima1,
          pacf_arima1,
          labels = c("ACF", "PACF"),
          nrow = 2, ncol = 1)

#-- Previsioni
pred_val1 <- forecast(mod1_arima,
                      h = 3504)   #-- Numero Validation
pred_val1 <- ts(pred_val1$mean,
                start = 585,
                frequency = 24)

#-- Grafico Previsioni
plot_val(pred_val1,
         "Previsioni ARIMA(2, 0, 0)(0, 1, 0)")

#-- MAE
mae_mod1_arima <- mean(abs(pred_val1 - validation_arima))
mae_mod1_arima

#--- 2. ARIMA(2, 0, 0)(0, 1, 1) ----
mod2_arima <- Arima(train_arima,
                    c(2, 0, 0),
                    c(0, 1, 1),
                    lambda = "auto",
                    include.constant = TRUE)
smod2_arima <- summary(mod2_arima)
smod2_arima[,"MAE"]

#-- ACF e PACF
acf_arima2 <- ggAcf(mod2_arima$residuals,
                    lag.max = 96,
                    main = "")
pacf_arima2 <- ggPacf(mod2_arima$residuals,
                      lag.max = 96,
                      main = "")

ggarrange(acf_arima2,
          pacf_arima2,
          labels = c("ACF", "PACF"),
          nrow = 2, ncol = 1)

#-- Previsioni
pred_val2 <- forecast(mod2_arima,
                      h = 3504)   #-- Numero Validation
pred_val2 <- ts(pred_val2$mean,
                start = 585,
                frequency = 24)

#-- Grafico Previsioni
plot_val(pred_val2,
         "Previsioni ARIMA(2, 0, 0)(0, 1, 1)")

#-- MAE
mae_mod2_arima <- mean(abs(pred_val2 - validation_arima))
mae_mod2_arima

#--- 3. ARIMA(3, 0, 0)(0, 1, 1) ----
mod3_arima <- Arima(train_arima,
                    c(3, 0, 0),
                    c(0, 1, 1),
                    lambda = "auto",
                    include.constant = TRUE)
smod3_arima <- summary(mod3_arima)
smod3_arima[,"MAE"]

#-- ACF e PACF
acf_arima3 <- ggAcf(mod3_arima$residuals,
                    lag.max = 96,
                    main = "")
pacf_arima3 <- ggPacf(mod3_arima$residuals,
                      lag.max = 96,
                      main = "")

ggarrange(acf_arima3,
          pacf_arima3,
          labels = c("ACF", "PACF"),
          nrow = 2, ncol = 1)

#-- Previsioni
pred_val3 <- forecast(mod3_arima,
                      h = 3504)   #-- Numero Validation
pred_val3 <- ts(pred_val3$mean,
                start = 585,
                frequency = 24)

#-- Grafico Previsioni
plot_val(pred_val3,
         "Previsioni ARIMA(3, 0, 0)(0, 1, 1)")

#-- MAE
mae_mod3_arima <- mean(abs(pred_val3 - validation_arima))
mae_mod3_arima

#--- 4. ARIMA(3, 0, 0)(0, 1, 2) ----
mod4_arima <- Arima(train_arima,
                    c(3, 0, 0),
                    c(0, 1, 2),
                    lambda = "auto",
                    include.constant = TRUE)
smod4_arima <- summary(mod4_arima)
smod4_arima[,"MAE"]

#-- ACF e PACF
acf_arima4 <- ggAcf(mod4_arima$residuals,
                    lag.max = 96,
                    main = "")
pacf_arima4 <- ggPacf(mod4_arima$residuals,
                      lag.max = 96,
                      main = "")

ggarrange(acf_arima4,
          pacf_arima4,
          labels = c("ACF", "PACF"),
          nrow = 2, ncol = 1)

#-- Previsioni
pred_val4 <- forecast(mod4_arima,
                      h = 3504)   #-- Numero Validation
pred_val4 <- ts(pred_val4$mean,
                start = 585,
                frequency = 24)

#-- Grafico Previsioni
plot_val(pred_val4,
         "Previsioni ARIMA(3, 0, 0)(0, 1, 2)")

#-- MAE
mae_mod4_arima <- mean(abs(pred_val4 - validation_arima))
mae_mod4_arima

#--- 5. ARIMA(3, 1, 2) con Regressione Fourier ----
y <- msts(train_arima,
          c(24, 365*24)) #-- Multistagionalità

mod5_arima <- Arima(y,
                    c(3, 1, 2),
                    lambda = "auto",
                    include.constant = TRUE,
                    xreg = fourier(y,
                                   K = c(2, 2)))

smod5_arima <- summary(mod5_arima)
smod5_arima[,"MAE"]

#-- ACF e PACF

#-- Previsioni
pred_val5 <- forecast(mod5_arima,
                      h = 3504,                   #-- Numero Validation
                      xreg = fourier(y,
                                     K = c(2, 2)))
pred_val5 <- ts(pred_val5$mean,
                start = 585,
                end = 731,
                frequency = 24)

#-- Grafico Previsioni
plot_val(pred_val5,
         "Previsioni ARIMA(3, 1, 2) con Regressione Fourier")

#-- MAE
mae_mod5_arima <- mean(abs(pred_val5 - validation_arima))
mae_mod5_arima

#-- 6. Auto-Arima: ARIMA(5, 0, 2)(2, 1, 0) ----
automod_arima <- auto.arima(train_arima,
                            lambda = "auto")
sautomod_arima <- summary(automod_arima)
sautomod_arima[, "MAE"]

#-- ACF e PACF
acf_automod <- ggAcf(automod_arima$residuals,
                     lag.max = 96,
                     main = "")
pacf_automod <- ggPacf(automod_arima$residuals,
                       lag.max = 96,
                       main = "")

ggarrange(acf_automod,
          pacf_automod,
          labels = c("ACF", "PACF"),
          nrow = 2, ncol = 1)

#-- Previsioni
pred_val_auto <- forecast(automod_arima,
                          h = 3504)
pred_val_auto <- ts(pred_val_auto$mean,
                    start = 585,
                    frequency = 24)

#-- Grafico Previsioni
plot_val(pred_val_auto,
         "Previsioni ARIMA(5, 0, 2)(2, 1, 0)")

#-- MAE
mae_automod_arima <- mean(abs(pred_val_auto - validation_arima))
mae_automod_arima

#-- Previsione Test Modello Migliore (ARIMA) ----
mae_arima_val <- rbind(mae_mod1_arima,
                       mae_mod2_arima,
                       mae_mod3_arima,
                       mae_mod4_arima,
                       mae_mod5_arima)
mae_arima_train <- rbind(smod1_arima[,"MAE"],
                         smod2_arima[,"MAE"],
                         smod3_arima[,"MAE"],
                         smod4_arima[,"MAE"],
                         smod5_arima[,"MAE"])

mae_arima <- cbind(mae_arima_train,
                   mae_arima_val)

colnames(mae_arima) <- c("Train", "Validation")
rownames(mae_arima) <- c("ARIMA(2,0,0)(0,1,0)",
                         "ARIMA(2,0,0)(0,1,1)",
                         "ARIMA(3,0,0)(0,1,1)",
                         "ARIMA(3,0,0)(0,1,2)",
                         "ARIMAX(3,1,2) Fourier")
mae_arima

#-- Confronto MAE sul Validation
y_best <- msts(ts_stand,
               c(24, 365*24)) #-- Multistagionalità

best_arima <- Arima(y_best,
                    c(3, 1, 2),
                    lambda = "auto",
                    include.constant = TRUE,
                    xreg = fourier(y_best,
                                   K = c(2, 2)))
sbest_arima <- summary(best_arima)
sbest_arima[,"MAE"]

#-- Previsioni
pred_best_arima <- forecast(best_arima,
                            h = 1464,
                            xreg = fourier(y_best,
                                           K = c(2, 2)))
pred_best_arima <- unscale(as.numeric(pred_best_arima$mean),
                           ts_stand)

#-- Aggiunta Date Test Set
fore <- c()
for (j in c(1:61)){
  for (i in c(1:24)){
    
    date <- as_date("2020-08-31") + j
    hours <- hms::hms(hours = i)
    
    print(ymd_hms(paste(date, " ", hours)))
    fore <- append(fore, ymd_hms(paste(date, " ", hours)))
  }
}

dyn_pred_best_arima <- xts::xts(x = pred_best_arima[1:(61*24)],
                                order.by = fore)

dyn_pred_val5 <- xts::xts(x = unscale(pred_val5[1:3504], ts_stand),
                          order.by = ymdhms[(nrow(ts) - 3503):nrow(ts)])

all_arima <- cbind(dyn_ts,
                   dyn_pred_val5,
                   dyn_pred_best_arima)/1000000
colnames(all_arima) <- c("Reali",
                         "Validation",
                         "Previsione")

dygraph(all_arima)                           %>%
  dySeries("Reali",
           label = "Reali")                  %>% #-- Train Dataset
  dySeries("Previsione",
           label = "Previsione")             %>% #-- Test Dataset
  dySeries("Validation",
           label = "Validation")             %>% #-- Validation Dataset
  dyOptions(drawPoints = TRUE,                   #-- Draw Points
            pointSize = 0.5,                     #-- Size of Points
            colors = c("blue",
                       "red",
                       "black"))             %>% #-- Colori
  dyAxis("y", label = "Valore (in Milioni)") %>% #-- Asse Y
  dyAxis("x", label = "Periodo",
         drawGrid = FALSE)                   %>% #-- Asse X
  dyRangeSelector(height = 20)               %>% #-- Slider 
  dyLegend(width = 350)                  

#-- UCM ----

#-- Scomposizione Train e Validation ----
train_ucm <- ts(ts_stand[1:as.integer(nrow(ts_stand)*0.8), ],
                frequency = 24,
                start = 1)

validation_ucm <- ts(ts_stand[(1 + as.integer(nrow(ts_stand)*0.8)) : nrow(ts_stand), ],
                     frequency = 24,
                     start = 585)

v_train_ucm <- var(train_ucm) #-- Varianza per Modelli

#-- 1. Local Linear Trend ----
mod1_ucm <- SSModel(train_ucm ~ SSMtrend(2,                    #-- 2 Equazioni da Calcolare
                                         Q = list(matrix(NA),
                                                  matrix(0))), #-- Matrice Q con Dimensioni 2*2
                    H = matrix(NA))                            #-- matrice H

mod1_ucm$Q
mod1_ucm$T
mod1_ucm$Z

#-- Adattamento del Modello
fit1_ucm <- fitSSM(mod1_ucm,
                   inits = log(c(v_train_ucm/10,    #-- Inizializzaione
                                 v_train_ucm/20,    #-- Inserire i valori
                                 v_train_ucm/100))) #-- della Varianza

fit1_ucm$optim.out$convergence                      #-- Convergenza
fit1_ucm$model$Q                                    #-- Varianze Stimate dal modello          

#-- Previsioni del Modello (Validation)
pred1_ucm <- predict(fit1_ucm$model,
                     n.ahead = length(validation_ucm))  #-- Numerosità Validation (3504)
pred1_ucm <- ts(pred1_ucm,                              #-- Dataset Previsione
                start = 585,                            #-- 585 Giorni
                frequency = 24)                         #-- Frequenza Giornaliera

#-- Grafico Previsioni
plot_val(pred1_ucm,
         "Previsioni Local Linear Trend")

#-- MAE
mae1_ucm <- mean(abs(validation_ucm - pred1_ucm))
mae1_ucm

#-- 2. Local Linear Trend con Stagionalità Dummy ----
mod2_ucm <- SSModel(train_ucm ~ SSMtrend(2,                     #-- 2 Equazioni da Calcolare
                                         Q = list(matrix(NA),
                                                  matrix(0))) + #-- Matrice Q con Dimensioni 2*2
                      SSMseasonal(24,                           #-- Periodo Stagionalità
                                  Q = matrix(NA),
                                  sea.type = 'dummy'),          #-- Dummy
                    H = matrix(NA))                             #-- Matrice H

fit2_ucm<- fitSSM(mod2_ucm, inits = log(c(v_train_ucm/10,
                                          v_train_ucm/100,      #-- Inizializzaione
                                          v_train_ucm/2,        #-- Inserire i valori
                                          v_train_ucm/50)))     #-- della Varianza

fit2_ucm$optim.out$convergence                                  #-- Convergenza
fit2_ucm$model$Q                                                #-- Varianze Stimate dal modello

#-- Previsioni del Modello (Validation)
pred2_ucm <- predict(fit2_ucm$model,
                     n.ahead = length(validation_ucm)) #-- Numerosità Validation (3504)
pred2_ucm <- ts(pred2_ucm,                             #-- Dataset Previsione
                start = 585,                           #-- 585 Giorni
                frequency = 24)                        #-- Frequenza Giornaliera

#-- Grafico Previsioni
plot_val(pred2_ucm,
         "Previsioni Local Linear Trend con Stagionalità Giornaliera")

#-- Focus Prima ed Ultima Settimana

#-- MAE
mae2_ucm <- mean(abs(validation_ucm - pred2_ucm))
mae2_ucm

#-- 3. Local Linear Trend con Stagionalità Trigonometrica ----
mod3_ucm <- SSModel(train_ucm ~ SSMtrend(2,                     #-- 2 Equazioni da Calcolare
                                         Q = list(matrix(NA),   
                                                  matrix(0))) + #-- Matrice Q con Dimensioni 2*2
                      SSMseasonal(24,                           #-- Periodo Stagionalità
                                  Q = matrix(NA),            
                                  sea.type = "trigonometric"),  #-- Trigonometrica
                    H = matrix(NA))                             #-- Matrice H

#-- Funzione di Update: da 1 a 4 le varianze non sono costanti,
#-- da 5 a 25 le varianzi sono costanti
updt3 <- function(pars, model){
  model$Q[1, 1, 1] <- exp(pars[1])
  model$Q[2, 2, 1] <- exp(pars[2])
  model$Q[3, 3, 1] <- exp(pars[3])
  model$Q[4, 4, 1] <- exp(pars[4])
  diag(model$Q[5 : 25, 5 : 25, 1]) <- exp(pars[5])
  model$H[1, 1, 1] <- exp(pars[6])
  model
}

fit3_ucm <- fitSSM(mod3_ucm,
                   log(c(v_train_ucm/10,
                         v_train_ucm/100,
                         v_train_ucm/75,
                         v_train_ucm/20,         #-- Inizializzaione
                         v_train_ucm/50,         #-- Inserire i valori
                         v_train_ucm/5)),        #-- della Varianza
                   updt3,                        #-- Funzione di Update
                   control = list(maxit = 1000)) #-- Numero Masimo di Iterazioni

fit3_ucm$optim.out$convergence                   #-- Convergenza
fit3_ucm$optim.out$counts                        #-- Varianze Stimate dal modello

#-- Previsioni del Modello (Validation)
pred3_ucm <- predict(fit3_ucm$model,
                     n.ahead = length(validation_ucm)) #-- Numerosità Validation (3504)
pred3_ucm <- ts(pred3_ucm,                             #-- Dataset Previsione
                start = 585,                           #-- 585 Giorni
                frequency = 24)                        #-- Frequenza Giornaliera

#-- Grafico Previsioni
plot_val(pred3_ucm,
         "Previsioni Local Linear Trend con Stagionalità Giornaliera (Trigonometrica)")

#-- MAE
mae3_ucm <- mean(abs(validation_ucm - pred3_ucm))
mae3_ucm

#-- 4. Local Linear Trend con Stagionalità Dumme e Ciclo Annuale ----
mod4_ucm <- SSModel(train_ucm ~ SSMtrend(2,                     #-- 2 Equazioni da Calcolare
                                         Q = list(matrix(NA),
                                                  matrix(0))) + #-- Matrice Q con Dimensioni 2*2
                      SSMseasonal(24,                           #-- Periodo Stagionalità
                                  Q = matrix(NA),
                                  sea.type = 'dummy')         + #-- Dummy
                      SSMcycle(24*365),                         #-- Ciclo Annuale
                    H = matrix(NA))                             #-- Matrice H

fit4_ucm <- fitSSM(mod4_ucm,
                  inits = log(c(v_train_ucm/10,
                                v_train_ucm/100,  #-- Inizializzaione
                                v_train_ucm/2,    #-- Inserire i valori
                                v_train_ucm/50))) #-- della Varianza

fit4_ucm$optim.out$convergence                    #-- Convergenza
fit4_ucm$model$Q                                  #-- Varianze Stimate dal modello

#-- Previsioni del Modello (Validation)
pred4_ucm <- predict(fit4_ucm$model,
                     n.ahead = length(validation_ucm)) #-- Numerosità Validation (3504)
pred4_ucm <- ts(pred4_ucm,                             #-- Dataset Previsione
                start = 585,                           #-- 585 Giorni
                frequency = 24)                        #-- Frequenza Giornaliera

#-- Grafico Previsioni
plot_val(pred4_ucm,
         "Previsioni LLT con Stagionalità Giornaliera (Dummy) e Ciclo Annuale")

#-- MAE
mae4_ucm <- mean(abs(validation_ucm - pred4_ucm))
mae4_ucm

#-- 5. Random Walk con Stagionalità Dummy e Ciclo Annuale ----
mod5_ucm <- SSModel(train_ucm ~ SSMtrend(1,             #-- 1 Equazione da Calcolare
                                         NA)          +
                      SSMseasonal(24,                   #-- Periodo Stagionalità
                                  NA,
                                  sea.type = "dummy") + #-- Dummy
                      SSMcycle(24*365),                 #-- #-- Ciclo Annuale
                    H = matrix(NA))                     #-- Matrice H

fit5_ucm <- fitSSM(mod5_ucm,
                  inits = log(c(v_train_ucm/10,
                                v_train_ucm/100,  #-- Inizializzaione
                                v_train_ucm/2,    #-- Inserire i valori
                                v_train_ucm/50))) #-- della Varianza

fit5_ucm$optim.out$convergence                    #-- Convergenza
fit5_ucm$model$Q                                  #-- Varianze Stimate dal modello

#-- Previsioni del Modello (Validation)
pred5_ucm <- predict(fit5_ucm$model,
                     n.ahead = length(validation_ucm)) #-- Numerosità Validation (3504)
pred5_ucm <- ts(pred5_ucm,                             #-- Dataset Previsione
                start = 585,                           #-- 585 Giorni
                frequency = 24)                        #-- Frequenza Giornaliera

#-- Grafico Previsioni
plot_val(pred5_ucm,
         "Previsioni Random Walk con Stagionalità Giornaliera (Dummy) e Ciclo Annuale")

#-- MAE
mae5_ucm <- mean(abs(validation_ucm - pred5_ucm))
mae5_ucm

#-- Previsione Test Modello Migliore (UCM) ----
mae_ucm_train <- rbind(mean(abs(unscale(fitted(fit1_ucm$model), ts_stand) - unscale(train_ucm, ts_stand))),
                       mean(abs(unscale(fitted(fit2_ucm$model), ts_stand) - unscale(train_ucm, ts_stand))),
                       mean(abs(unscale(fitted(fit3_ucm$model), ts_stand) - unscale(train_ucm, ts_stand))),
                       mean(abs(unscale(fitted(fit4_ucm$model), ts_stand) - unscale(train_ucm, ts_stand))),
                       mean(abs(unscale(fitted(fit5_ucm$model), ts_stand) - unscale(train_ucm, ts_stand))))
mae_ucm_val <- rbind(mae1_ucm,
                     mae2_ucm,
                     mae3_ucm,
                     mae4_ucm,
                     mae5_ucm)

mae_ucm <- cbind(mae_ucm_train,
                 mae_ucm_val)

colnames(mae_ucm) <- c("Train", "Validation")
rownames(mae_ucm) <- c("LLT",
                       "LLT + Sd",
                       "LLT + St",
                       "LLT + Sd + Ca",
                       "RW + Sd + Ca")
mae_ucm

pred_best_ucm <- SSModel(ts_stand ~ SSMtrend(1,         #-- 1 Equazione da Calcolare
                                             NA)      +
                      SSMseasonal(24,                   #-- Periodo Stagionalità
                                  NA,
                                  sea.type = "dummy") + #-- Dummy
                      SSMcycle(24*365),                 #-- Ciclo Annuale
                    H = matrix(NA))                     #-- Matrice H

fit_best_ucm <- fitSSM(pred_best_ucm,
                       inits = log(c(v_train_ucm/10,
                                     v_train_ucm/100,  #-- Inizializzaione
                                     v_train_ucm/2,    #-- Inserire i valori
                                     v_train_ucm/50))) #-- della Varianza

fit_best_ucm$optim.out$convergence                     #-- Convergenza
fit_best_ucm$model$Q                                   #-- Varianze Stimate dal modello

#-- Previsioni
pred_best_ucm <- predict(fit_best_ucm$model,
                         n.ahead = 1464)            #-- Numerosità Test
pred_best_ucm <- unscale(as.numeric(pred_best_ucm), #-- Valori non Scalati
                           ts_stand)

dyn_pred_best_ucm <- xts::xts(x = pred_best_ucm,
                              order.by = fore)

dyn_pred_val5 <- xts::xts(x = unscale(pred5_ucm, ts_stand),
                          order.by = ymdhms[(nrow(ts) - 3503):nrow(ts)])

all_ucm <- cbind(dyn_ts,
                 dyn_pred_val5,
                 dyn_pred_best_ucm)/1000000
colnames(all_ucm) <- c("Reali",
                       "Validation",
                       "Previsione")

dygraph(all_ucm)                             %>%
  dySeries("Reali",
           label = "Reali")                  %>% #-- Train Dataset
  dySeries("Previsione",
           label = "Previsione")             %>% #-- Test Dataset
  dySeries("Validation",
           label = "Validation")             %>% #-- Validation Dataset
  dyOptions(drawPoints = TRUE,                   #-- Draw Points
            pointSize = 0.5,                     #-- Size of Points
            colors = c("blue",
                       "red",
                       "black"))             %>% #-- Colori
  dyAxis("y", label = "Valore (in Milioni)") %>% #-- Asse Y
  dyAxis("x", label = "Periodo",
         drawGrid = FALSE)                   %>% #-- Asse X
  dyRangeSelector(height = 20)               %>% #-- Slider 
  dyLegend(width = 350)                  

#-- Machine Learning ----

#-- KNN ----
#-- Scomposizione Train e Validation ----
train_knn <- ts_stand[1:as.integer(nrow(ts)*0.8)]
validation_knn <- ts_stand[(1 + as.integer(nrow(ts)*0.8)) : nrow(ts)]

#-- 1. K-NN (1 Settimana Lookback) ----

#-- Parametri Modello
p_week <- 1:(24*7)      #-- Lunghezza del Frammento di Serie Storica di cui Individuare i k Vicini 
k      <- seq(1, 15, 2) #-- Numero di Vicini da Individuare
h      <- 3504          #-- Previsione (Numero Validation)

for (el in k){
  mod_knn_week <- knn_forecasting(timeS = train_knn,  #-- Serie Storica
                                  h    = h,           #-- Orizzonte
                                  lags = p_week,      #-- Autoregressione
                                  k    = el,          #-- Numero di k-NN
                                  msas = "recursive", #-- Recursive
                                  cf   = "mean")      #-- Combinazione Futuri
  
  #-- Previsione
  print(el)
  pred_knn <- ts(mod_knn_week$prediction, #-- Dataset Previsione
                 start = 585,             #-- 585 Giorni
                 frequency = 24)          #-- Frequenza Giornaliera
  
  #-- MAE
  mae_knn <- mean(abs(validation_knn - pred_knn))
  print(mae_knn)
}

#-- Modello Migliore (k = 9)
mod_knn_week <- knn_forecasting(timeS = train_knn,  #-- Serie Storica
                                h    = h,           #-- Orizzonte
                                lags = p_week,      #-- Autoregressione
                                k    = 9,           #-- Numero di k-NN
                                msas = "recursive", #-- Recursive
                                cf   = "mean")      #-- Combinazione Futuri

pred_knn <- ts(mod_knn_week$prediction, #-- Dataset Previsione
               start = 585,             #-- 585 Giorni
               frequency = 24)          #-- Frequenza Giornaliera

#-- MAE
mae_knn_week <- mean(abs(validation_knn - pred_knn_week))
print(mae_knn_week)

mae_knn_week_train <- predict(mod_knn_week$model$ts)
mae_knn_week_train <- mean(abs(mae_knn_week_train$fitted - train_knn))

#-- Grafico Previsioni
plot_val(pred_knn,
         "Previsioni K-NN con Lookback di Ricerca Settimanale")

#-- 2. K-NN (1 Giorno Lookback) ----

#-- Parametri Modello
p_day <- 1:24
k     <- seq(1, 30, 2)

for (el in k){
  mod_knn_day <- knn_forecasting(timeS = train_knn,   #-- Serie Storica
                                 h     = h,           #-- Orizzonte
                                 lags  = p_day,       #-- Autoregressione
                                 k     = el,          #-- Numero di k-NN
                                 msas  = "recursive", #-- Recursive
                                 cf    = "mean")      #-- Combinazione Futuri
  
  #-- Previsione
  print(el)
  pred_knn <- ts(mod_knn_day$prediction, #-- Dataset Previsione
                 start = 585,            #-- 585 Giorni
                 frequency = 24)         #-- Frequenza Giornaliera
  
  #-- MAE
  mae_knn <- mean(abs(validation_knn - pred_knn))
  print(mae_knn)
}

#-- Modello Migliore (k = 15)
mod_knn_day <- knn_forecasting(timeS = train_knn,  #-- Serie Storica
                               h    = h,           #-- Orizzonte
                               lags = p_week,      #-- Autoregressione
                               k    = 15,          #-- Numero di k-NN
                               msas = "recursive", #-- Recursive
                               cf   = "mean")      #-- Combinazione Futuri

pred_knn_day <- ts(mod_knn_week$prediction, #-- Dataset Previsione
                   start = 585,             #-- 585 Giorni
                   frequency = 24)          #-- Frequenza Giornaliera

#-- MAE
mae_knn_day <- mean(abs(validation_knn - pred_knn_day))
print(mae_knn_day)

mae_knn_day_train <- predict(mod_knn_day$model$ts)
mae_knn_day_train <- mean(abs(mae_knn_day_train$fitted - train_knn))

#-- Grafico Previsioni
plot_val(pred_knn_day,
         "Previsioni K-NN con Lookback di Ricerca Giornaliera")

#-- RNN ----

#-- Parametri Modello
datalags   <- 24*7 #-- Orizzonte di ricerca
batch.size <- 73
epochs     <- 10

#-- Scomposizione Train e Validation ----
train_lstm <- as.data.frame(ts_stand[seq(16352 + datalags)])
validation_lstm <- as.data.frame(ts_stand[16352 + datalags + seq(1168 + datalags)])

df_val <- ts_stand[(17520-1167) : 17520,] #-- estrarre per mae
df_train <- ts_stand[1:(17520-1168), ]

colnames(train_lstm) <- c("Valore")
colnames(validation_lstm) <- c("Valore")

x.train <- array(data = lag(train_lstm$Valore, datalags)[ -(1 : datalags)],
                 dim = c(nrow(train_lstm) - datalags, datalags, 1))

y.train <- array(data = train_lstm$Valore[-(1:datalags)],
                 dim = c(nrow(train_lstm)-datalags, 1))

x.test <- array(data = lag(validation_lstm$Valore,
                           datalags)[ -(1 : datalags)],
                dim = c(nrow(validation_lstm) - datalags, datalags, 1))

y.test <- array(data = validation_lstm$Valore[ -(1 : datalags)],
                dim = c(nrow(validation_lstm) - datalags, 1))

#-- 3. LSTM (1 Layer) ----

#-- Costruzione Modello
model_lstm1 <- keras_model_sequential() #-- Costruzione Strato per Strato
model_lstm1 %>%
  layer_lstm(units            = 16,             #-- Numero Neuroni 
             input_shape      = c(datalags, 1), #-- Dimensione Input
             batch_size       = batch.size,
             return_sequences = TRUE,
             stateful         = TRUE) %>%
  layer_dense(units = 1)                        #-- Layer Denso (Previsione)

#-- Compilazione
model_lstm1 %>%
  compile(loss = 'mae',                    #-- Funzione di Perdita
          optimizer = optimizer_rmsprop()) #-- Ottimizzatore

print(model_lstm1)

#-- Addestramento
history_lstm1 <- model_lstm1 %>%
  fit(x          = x.train,      #-- Train
      y          = y.train,      #-- Test
      epochs     = epochs,       #-- Epoche di Addestramento
      batch_size = batch.size,   #-- Batch Size
      verbose    = 0,
      shuffle    = FALSE)

#-- Funzione di Perdita
plot(history_lstm1)

#-- Previsione
pred_out_lstm1 <- model_lstm1 %>%
  predict(x.test,
          batch_size = batch.size)

pred_out_lstm1[, 168, 1]

#-- Grafico Previsione
autoplot(ts(df_train, start = 1),
         main = "Previsione LSTM (1 Layer)",    #-- Titolo
         xlab = "Periodo",                      #-- Label X
         ylab = "Valore (in Milioni)")        + #-- Label Y
  autolayer(ts(df_val, start = 16352),                   
            series = "Reale")                 + #-- Validation
  autolayer(ts(pred_out_lstm1[,1,1],
               start = (16352 + datalags)),
            series = "Previsione")            + #-- Previsione
  scale_y_continuous(breaks = c(-2, 0, 2, 4),   
                     labels = c(2, 4, 6, 8))  + #-- Interruzioni
  scale_color_manual(values = c("#00AFBB",
                                "#E7B800",
                                "#FC4E07"))   + #-- Colori
  labs(color = "Validation")                  + #-- Legenda
  theme_bw()                                    #-- Tema

#-- MAE
mae_lstm1 <- colMeans(drop_na(as.data.frame(abs(df_val - pred_out_lstm1[, 1, 1]))))
names(mae_lstm1) <- c("LSTM_1")
mae_lstm1

#-- 4. LSTM (2 Layer) ----

#-- Costruzione Modello
model_lstm2 <- keras_model_sequential()
model_lstm2 %>%
  layer_lstm(units            = 16,             #-- Numero Neuroni 
             input_shape      = c(datalags, 1), #-- Dimensione Input
             batch_size       = batch.size,
             return_sequences = TRUE,
             stateful         = TRUE) %>%       #-- Layer di tipo LSTM
  layer_dropout(rate = 0.5) %>%                 #-- Layer di Dropout
  layer_lstm(units            = 8,
             return_sequences = FALSE,
             stateful         = TRUE) %>%       #-- Layer di tipo LSTM
  layer_dropout(rate = 0.5) %>%                 #-- Layer di Dropout
  layer_dense(units = 1)                        #-- Layer Denso (Previsione)

#-- Compilazione
model_lstm2 %>%
  compile(loss = 'mae',       #-- Funzione di Perdita
          optimizer = 'adam') #-- Ottimizzatore

print(model_lstm2)

#-- Addestramento
history_lstm2 <- model_lstm2 %>%
  fit(x          = x.train,    #-- Train
      y          = y.train,    #-- Test
      epochs     = epochs,     #-- Epoche di Addestramento
      batch_size = batch.size,
      verbose    = 0,
      shuffle    = FALSE) 

#-- Funzione di Perdita
plot(history_lstm2)

#-- Previsione
pred_out_lstm2 <- model_lstm2 %>%
  predict(x.test,
          batch_size = batch.size)

#-- Grafico Previsione
autoplot(ts(df_train, start = 1),
         main = "Previsione LSTM (2 Layer)",    #-- Titolo
         xlab = "Periodo",                      #-- Label X
         ylab = "Valore (in Milioni)")        + #-- Label Y
  autolayer(ts(df_val, start = 16352),                   
            series = "Reale")                 + #-- Validation
  autolayer(ts(pred_out_lstm2,
               start = (16352 + datalags)),
            series = "Previsione")            + #-- Previsione
  scale_y_continuous(breaks = c(-2, 0, 2, 4),   
                     labels = c(2, 4, 6, 8))  + #-- Interruzioni
  scale_color_manual(values = c("#00AFBB",
                                "#E7B800",
                                "#FC4E07"))   + #-- Colori
  labs(color = "Validation")                  + #-- Legenda
  theme_bw()                                    #-- Tema

#-- MAE
mae_lstm2 <- colMeans(drop_na(as.data.frame(abs(df_val - pred_out_lstm2))))
names(mae_lstm2) <- c("LSTM_2")
mae_lstm2

#-- 5. GRU ----

#-- Costruzione Modello
mod_gru1 <- keras_model_sequential()
mod_gru1 %>%
  layer_gru(units             = 20,             #-- Numero Neuroni 
            input_shape       = c(datalags, 1), #-- Dimensione Input
            batch_size        = batch.size,
            dropout           = 0.3,
            recurrent_dropout = 0.5) %>%
  layer_dense(units      = 1,
              activation = "linear")

#-- Compilazione
mod_gru1 %>% 
  compile(loss = 'mae',       #-- Funzione di Perdita
          optimizer = 'adam') #-- Ottimizzatore

mod_gru1

#-- Addestramento
history_gru1 <- mod_gru1 %>%
  fit(x          = x.train,    #-- Train
      y          = y.train,    #-- Test
      epochs     = 25,         #-- Epoche di Addestramento
      batch_size = batch.size,
      verbose    = 0,
      shuffle    = FALSE)
plot(history_gru1)

#-- Previsione
pred_out_gru1 <- mod_gru1 %>%
  predict(x.test,
          batch_size = batch.size)

#-- Grafico Previsione
autoplot(ts(df_train, start = 1),
         main = "Previsione GRU",    #-- Titolo
         xlab = "Periodo",                      #-- Label X
         ylab = "Valore (in Milioni)")        + #-- Label Y
  autolayer(ts(df_val, start = 16352),                   
            series = "Reale")                 + #-- Validation
  autolayer(ts(pred_out_gru1,
               start = (16352 + datalags)),
            series = "Previsione")            + #-- Previsione
  scale_y_continuous(breaks = c(-2, 0, 2, 4),   
                     labels = c(2, 4, 6, 8))  + #-- Interruzioni
  scale_color_manual(values = c("#00AFBB",
                                "#E7B800",
                                "#FC4E07"))   + #-- Colori
  labs(color = "Validation")                  + #-- Legenda
  theme_bw()                                    #-- Tema

#-- MAE
mae_gru1 <- colMeans(drop_na(as.data.frame(abs(df_val - pred_out_gru1))))
names(mae_gru1) <- c("GRU_1")
mae_gru1

#-- Salvataggio Modelli ----
save_model_hdf5(model_lstm1, "model_lstm1.h5")
save_model_hdf5(model_lstm2, "model_lstm2.h5")
save_model_hdf5(mod_gru1, "model_gru1.h5")

#-- Previsione Test Modello Migliore (ML) ----

#-- Confronto MAE sul Validation
mae_ml_val <- rbind(mae_knn,     #-- K-NN Week
                    mae_knn_day, #-- K-NN Day
                    mae_lstm1,   #-- LSTM 1 Layer
                    mae_lstm2,   #-- LSTM 2 Layer
                    mae_gru1)    #-- GRU

mae_ml_train <- rbind(mae_knn_week_train,
                          mae_knn_day_train,
                          history_lstm1$metrics$loss[epochs],
                          history_lstm1$metrics$loss[epochs],
                          history_gru1$metrics$loss[25])

mae_ml <- cbind(mae_ml_train,
                mae_ml_val)

colnames(mae_ml) <- c("Train",
                      "Validation")
rownames(mae_ml) <- c("K-NN Week",
                      "K-NN Day",
                      "LSTM 1 Layer",
                      "LSTM 2 Layer",
                      "GRU")
mae_ml

#-- Scomposizione Train e Test
train_best_lstm <- as.data.frame(ts_stand[seq(17520 + datalags)])

x.train_best <- array(data = lag(train_lstm$Valore, datalags)[ -(1 : datalags)],
                      dim = c(nrow(train_best_lstm) - datalags, datalags, 1))

y.train_best <- array(data = train_lstm$Valore[-(1:datalags)],
                      dim = c(nrow(train_best_lstm)-datalags, 1))

#-- Costruzione Modello
model_best_ml <- keras_model_sequential()
model_best_ml %>%
  layer_lstm(units            = 16,             #-- Numero Neuroni 
             input_shape      = c(datalags, 1), #-- Dimensione Input
             batch_size       = batch.size,
             return_sequences = TRUE,
             stateful         = TRUE) %>%       #-- Layer di tipo LSTM
  layer_dropout(rate = 0.5) %>%                 #-- Layer di Dropout
  layer_lstm(units            = 8,
             return_sequences = FALSE,
             stateful         = TRUE) %>%       #-- Layer di tipo LSTM
  layer_dropout(rate = 0.5) %>%                 #-- Layer di Dropout
  layer_dense(units = 1)                        #-- Layer Denso (Previsione)

#-- Compilazione
model_best_ml %>%
  compile(loss = 'mae',       #-- Funzione di Perdita
          optimizer = 'adam') #-- Ottimizzatore

print(model_best_ml)

#-- Addestramento
history_best_ml <- model_best_ml %>%
  fit(x          = x.train_best,     #-- Train
      y          = y.train_best,     #-- Test
      epochs     = epochs,           #-- Epoche di Addestramento
      batch_size = batch.size,
      verbose    = 0,
      shuffle    = FALSE) 

#-- Funzione di Perdita
plot(history_best_ml)

save_model_hdf5(model_best_ml, "model_best.h5")

#-- Previsione
pred_list <- c(tail(ts_stand, n = datalags))

for (single_value in c(1: 1464)){
  single_value <- pred_list[(length(pred_list)-datalags):length(pred_list)]
  single_value <- array_reshape(single_value, c(1, 1, datalags))
  single_pred <- model_best_ml %>%
    predict(single_value)
  
  print(single_pred)
  pred_list <- append(single_pred)
}

dyn_pred_best_ml <- xts::xts(x = pred_list,
                             order.by = fore)

dyn_pred_val_lstm2 <- xts::xts(x = unscale(pred_out_lstm2, ts_stand)[1:1000],
                               order.by = (ymdhms[(nrow(ts) - 999):nrow(ts)]))

all_ml <- cbind(dyn_ts,
                dyn_pred_val_lstm2,
                dyn_pred_best_ml)/1000000
colnames(all_ml) <- c("Reali",
                      "Validation",
                      "Previsione")

dygraph(all_ml)                              %>%
  dySeries("Reali",
           label = "Reali")                  %>% #-- Train Dataset
  dySeries("Previsione",
           label = "Previsione")             %>% #-- Test Dataset
  dySeries("Validation",
           label = "Validation")             %>% #-- Validation Dataset
  dyOptions(drawPoints = TRUE,                   #-- Draw Points
            pointSize = 0.5,                     #-- Size of Points
            colors = c("blue",
                       "red",
                       "black"))             %>% #-- Colori
  dyAxis("y", label = "Valore (in Milioni)") %>% #-- Asse Y
  dyAxis("x", label = "Periodo",
         drawGrid = FALSE)                   %>% #-- Asse X
  dyRangeSelector(height = 20)               %>% #-- Slider 
  dyLegend(width = 350)                  

#-- Confronto Modelli Migliori ----
all <- cbind(dyn_pred_best_arima,
             dyn_pred_best_ucm,
             dyn_pred_best_ml)/1000000
colnames(all) <- c("ARIMA",
                   "UCM",
                   "ML")

dygraph(all)                                 %>%
  dySeries("ARIMA",
           label = "ARIMA")                  %>% #-- Train Dataset
  dySeries("UCM",
           label = "UCM")                    %>% #-- Test Dataset
  dySeries("ML",
           label = "ML")                     %>% #-- Validation Dataset
  dyOptions(drawPoints = TRUE,                   #-- Draw Points
            pointSize = 0.5,                     #-- Size of Points
            colors = c("blue",
                       "red",
                       "black"))             %>% #-- Colori
  dyAxis("y", label = "Valore (in Milioni)") %>% #-- Asse Y
  dyAxis("x", label = "Periodo",
         drawGrid = FALSE)                   %>% #-- Asse X
  dyRangeSelector(height = 20,
                  dateWindow = c("2020-09-01 1:00:00",
                                 "2020-09-08 1:00:00")) %>% #-- Slider 
  dyLegend(width = 350)

#-- Scrittura File ----
df_date <- c()
df_hours <- c()
for (j in c(1:61)){
  for (i in c(1:24)){
    
    date <- as_date("2020-09-01") + j
    hours <- i
    
    df_date <- append(df_date, date)
    df_hours <- append(df_hours, hours)
  }
}

test_data <- cbind(df_date,
                   df_hours,
                   pred_best_arima[1:(61*24)],
                   pred_best_ucm,
                   pred_out_best_ml)

colnames(test_data) <- c("Data", "Ora",
                         "ARIMA", "UCM", "ML")

write_csv2(as.data.frame(test_data),
           "G:\\Il mio Drive\\Università\\Data Science\\2° Anno\\Streaming Data Management and Time Series Analysis\\Esame\\calendar.csv")
