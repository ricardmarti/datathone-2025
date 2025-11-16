import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
import os
os.chdir("/Users/ricard/Desktop/ap1/datathone-2025")

# carregar dades
train = pd.read_csv("/Users/ricard/Desktop/ap1/datathone-2025/train.csv", sep=';')
test = pd.read_csv("/Users/ricard/Desktop/ap1/datathone-2025/test.csv", sep=';')

# Definir target
target = "weekly_demand"

def trobar_columnes_valides_millorat(train_df, test_df, target):
    
    columnes_comunes = list(set(train_df.columns) & set(test_df.columns))
    
    columnes_valides = []
    
    for col in columnes_comunes:
        if col == target or col == "ID":
            continue
            
        # fusió
        train_null_ratio = train_df[col].isnull().mean()
        test_null_ratio = test_df[col].isnull().mean()
        
        # fusió
        if train_null_ratio < 0.7 and test_null_ratio < 0.7:
            # IA
            if train_df[col].dtype in ['int64', 'float64']:
                if train_df[col].std() > 0: 
                    columnes_valides.append(col)
            else:
                # per categòriques, verificar que tingui diversitat
                if train_df[col].nunique() > 1:
                    columnes_valides.append(col)
    
    print(f"IA selecciona {len(columnes_valides)} columnes vàlides")
    return columnes_valides


columnes_valides = trobar_columnes_valides_millorat(train, test, target)

# preparar dades
def preparar_dades_sense_buits(df, columnes):
    df_prep = df[columnes].copy()
    
    for col in df_prep.columns:
        if df_prep[col].dtype in ['int64', 'float64']:
            # per numériques, la mitjana
            df_prep[col] = df_prep[col].fillna(df_prep[col].median())
        else:
            #per categòriques, moda o unknown
            if df_prep[col].notna().any():
                df_prep[col] = df_prep[col].fillna(df_prep[col].mode()[0] if not df_prep[col].mode().empty else 'Unknown')
            else:
                df_prep[col] = 'Unknown'
    
    # Convertir categòriques o numèriques
    categorical_cols = df_prep.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        df_prep[col] = df_prep[col].astype('category').cat.codes
    
    return df_prep

# Preparar dades
X_train = preparar_dades_sense_buits(train, columnes_valides)
X_test = preparar_dades_sense_buits(test, columnes_valides)
y_train = train[target]

# fusió: 
def calcular_factor_escalat_agressiu(X_train, y_train):
    # FUSIÓ
    factor_base = 3.0 
    
    # IA
    if 'num_stores' in X_train.columns:
        avg_stores = X_train['num_stores'].mean()
        if avg_stores > 80: 
            factor_base *= 1.3 
    
    if 'price' in X_train.columns:
        avg_price = X_train['price'].mean()
        if avg_price < 50:  
            factor_base *= 1.4  
    
    print(f"factor d'escalat aplicat: {factor_base:.2f}")
    return factor_base

factor_agressiu = calcular_factor_escalat_agressiu(X_train, y_train)
y_train_escalat = y_train * factor_agressiu

# FUSIÓ
def crear_modelo_optimizado(X_train, y_train_escalat):

    model_base = RandomForestRegressor(
        n_estimators=100,
        max_depth=20,
        random_state=42,
        n_jobs=-1
    )
    
    # IA
    try:
        scores = cross_val_score(model_base, X_train, y_train_escalat, 
                               cv=3, scoring='r2', n_jobs=-1)
        print(f"validació IA - R² promig: {scores.mean():.4f}")
    except:
        print("validació IA - usant model base")
    
    return model_base

model = crear_modelo_optimizado(X_train, y_train_escalat)

# Entrenar model
model.fit(X_train, y_train_escalat)

# FUSIÓ
demand_predictions = model.predict(X_test)

# IA
percentils = np.percentile(demand_predictions, [10, 25, 50, 75, 90])
print(f"anàlisi IA de prediccions:")

# FUSIÓ
def calcular_produccion_masiva(predictions, X_test):
    production_units = np.zeros(len(predictions), dtype=int)
    
    for i, pred in enumerate(predictions):
        # FUSIÓ
        if pred < 1500:  
            production_units[i] = pred * 3.0  
        elif pred < 3000:
            production_units[i] = pred * 2.5 
        else:
            production_units[i] = pred * 2.0
        
        # FUSIÓ
        production_units[i] = max(production_units[i], 800)
        
        # IA
        if 'num_stores' in X_test.columns:
            stores = X_test.iloc[i]['num_stores']
            if stores > 80:  
                production_units[i] = int(np.ceil(production_units[i] * 1.15))  
        
        if 'price' in X_test.columns:
            price = X_test.iloc[i]['price']
            if price < 60:  
                production_units[i] = int(np.ceil(production_units[i] * 1.10))  
    
    return production_units

production_units = calcular_produccion_masiva(demand_predictions, X_test)

# FUSIÓ
production_units = production_units + 300  
production_units = np.ceil(production_units).astype(int)

# FUSIÓ
production_units = np.where(production_units < 1500, 1500, production_units) 

# IA


# document de
production_document = pd.DataFrame({
    'ID': test['ID'],
    'Production': production_units
})

production_document.to_csv('data-mango.csv', index=False)
print(f"arxiu guardat com a data-mango.csv")