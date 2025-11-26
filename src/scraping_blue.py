# %%
import requests
import pandas as pd
from datetime import datetime, timedelta, timezone
import time
import os

# %%

PAIR = {"asset": "USDT", "fiat": "ARS"}
INTERVAL_MINUTES = 5    # cada cuántos minutos actualizar
ROWS = 20               # cantidad de anuncios por llamada
OUTPUT_FILE = "usdt_ars_intraday.csv"
DURATION_HOURS = 24     # duración total de la recolección en horas

# %%
def get_binance_p2p(trade_type="SELL"):
    url = "https://p2p.binance.com/bapi/c2c/v2/friendly/c2c/adv/search"
    payload = {
        "asset": PAIR["asset"],
        "fiat": PAIR["fiat"],
        "merchantCheck": False,
        "page": 1,
        "rows": ROWS,
        "tradeType": trade_type,
    }
    response = requests.post(url, json=payload)
    data = response.json()
    ads = data.get("data", [])
    results = []
    for ad in ads:
        adv = ad["adv"]
        # Manejar métodos de pago que puedan ser None
        trade_methods = []
        for m in adv["tradeMethods"]:
            method_name = m.get("tradeMethodName")
            if method_name is not None:
                trade_methods.append(str(method_name))

        results.append({
            "timestamp": datetime.now(timezone.utc).isoformat(),  # Corregido deprecated
            "tradeType": trade_type,
            "price": float(adv["price"]),
            "available_amount": float(adv["surplusAmount"]),
            "min_single_trans": float(adv["minSingleTransAmount"]),
            "max_single_trans": float(adv["dynamicMaxSingleTransAmount"]),
            "payment_methods": ", ".join(trade_methods) if trade_methods else "Unknown",
            "nickName": ad["advertiser"]["nickName"],
        })
    return pd.DataFrame(results)


# --- FUNCIÓN PARA REGISTRAR DATOS ---
def collect_snapshot():
    try:
        df_sell = get_binance_p2p("SELL")
        df_buy = get_binance_p2p("BUY")
        df = pd.concat([df_sell, df_buy])

        file_exists = os.path.isfile(OUTPUT_FILE)  # verifica si el archivo existe

        df.to_csv(
            OUTPUT_FILE,
            mode="a",
            index=False,
            header=not file_exists  # escribe encabezado solo la primera vez
        )
        print(f"{datetime.now().isoformat()} -> {len(df)} registros guardados.")
        return True
    except Exception as e:
        print(f"Error en collect_snapshot: {e}")
        return False

# Función para mostrar progreso
def show_progress(start_time, end_time):
    current_time = datetime.now()
    elapsed = current_time - start_time
    total = end_time - start_time
    progress = (elapsed / total) * 100
    remaining = end_time - current_time
    print(f"Progreso: {progress:.1f}% - Tiempo restante: {str(remaining).split('.')[0]}")

# %%
print("Recolectando datos de Binance P2P (USDT–ARS) por 24 horas...")
start_time = datetime.now()
end_time = start_time + timedelta(hours=DURATION_HOURS)

print(f"Inicio: {start_time.isoformat()}")
print(f"Fin programado: {end_time.isoformat()}")

successful_collections = 0
failed_collections = 0

while datetime.now() < end_time:
    if collect_snapshot():
        successful_collections += 1
    else:
        failed_collections += 1

    show_progress(start_time, end_time)

    # Verificar si el siguiente ciclo superaría el tiempo límite
    current_time = datetime.now()
    time_until_next = INTERVAL_MINUTES * 60
    if current_time + timedelta(seconds=time_until_next) > end_time:
        time_remaining = (end_time - current_time).total_seconds()
        if time_remaining > 0:
            print(f"Esperando {time_remaining/60:.1f} minutos finales...")
            time.sleep(time_remaining)
        break

    time.sleep(time_until_next)

print("\n" + "="*50)
print("Recolección de 24 horas completada.")
print(f"Archivo guardado: {OUTPUT_FILE}")
print(f"Recolecciones exitosas: {successful_collections}")
print(f"Recolecciones fallidas: {failed_collections}")
print(f"Tasa de éxito: {(successful_collections/(successful_collections+failed_collections))*100:.1f}%")
