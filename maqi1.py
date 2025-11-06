import pandas as pd
import numpy as np
import requests
import time
from datetime import datetime, timedelta
from prophet import Prophet
from xgboost import XGBRegressor
from fpdf import FPDF
import matplotlib.pyplot as plt
import os
import logging
from sklearn.metrics import mean_squared_error, mean_absolute_error
import math
import warnings
from scipy.stats import pearsonr
import seaborn as sns

# Set up plotting and logging
plt.switch_backend('Agg')
warnings.filterwarnings("ignore")
sns.set_style("whitegrid")

# Configuration
DATA_DIR = "/home/jeyadev/reports/"
os.makedirs(DATA_DIR, exist_ok=True)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Constants
STATIONS = {
    "Alandur Bus Depot": (13.0025, 80.2061),
    "Kodungaiyur": (13.1580, 80.2541),
    "Manali": (13.1785, 80.2599),
    "Perungudi": (12.9700, 80.2400),
    "Royapuram": (13.1075, 80.2958),
    "Velachery": (12.9855, 80.2185)
}

POLLUTANTS = ["pm10", "pm2_5", "no2", "so2", "o3", "co"]
WEATHER_PARAMS = ["temperature", "humidity", "wind_speed"]
MODEL_COLORS = {'prophet': '#1f77b4', 'xgb': '#ff7f0e'}
AQI_GRADIENT = [(0, 153, 102), (255, 222, 51), (255, 153, 51), (204, 0, 51), (102, 0, 153)]
LAG_DAYS = 7
MIN_TRAINING_SAMPLES = 30
MORTALITY_PARAMS = ['pm25', 'pm10', 'no2', 'so2', 'o3']
AQI_THRESHOLDS = [50, 100, 150, 200, 300]
AQI_STATUSES = ["Good", "Moderate", "Unhealthy", "Very Unhealthy", "Hazardous"]

class ModernReport(FPDF):
    """Custom FPDF class for generating modern air quality and health reports."""
    def __init__(self, station):
        super().__init__(orientation='L')
        self.station = station
        self.set_auto_page_break(auto=True, margin=15)
        self.set_margins(20, 25, 20)

    def header(self):
        self.set_font('Helvetica', 'B', 18)
        self.set_text_color(57, 96, 156)
        self.cell(0, 10, f"AIR QUALITY & HEALTH REPORT - {self.station.upper()}", 0, 1, 'C')
        self.set_font('Helvetica', '', 10)
        self.cell(0, 6, f"Generated on {datetime.now().strftime('%d %b %Y %H:%M')}", 0, 1, 'C')
        self.ln(10)

    def footer(self):
        self.set_y(-15)
        self.set_font('Helvetica', '', 8)
        self.set_text_color(100, 100, 100)
        self.cell(0, 10, f"Page {self.page_no()}", 0, 0, 'C')

    def _section_title(self, title):
        self.set_font('Helvetica', 'B', 14)
        self.set_fill_color(57, 96, 156)
        self.set_text_color(255, 255, 255)
        self.cell(0, 8, title, 0, 1, 'L', fill=True)
        self.ln(4)

    def add_executive_summary(self, summary):
        self._section_title("Executive Summary")
        self.set_font('Helvetica', '', 10)
        self.set_text_color(40, 40, 40)

        card_width = (self.w - 50) / 3
        metrics = [
            ("Current AQI", f"{summary['current_aqi']:.0f} ({summary['aqi_status']})", AQI_GRADIENT[summary['aqi_level']]),
            ("Dominant Pollutant", f"{summary['dominant_pollutant'].upper()}\n{summary['dominant_value']:.1f} μg/m³", (57, 96, 156)),
            ("Health Risk", f"{summary['risk_level']}\n{summary['risk_change']}", (234, 67, 53) if 'High' in summary['risk_level'] else (251, 188, 5))
        ]

        for i, (label, value, color) in enumerate(metrics):
            self.set_xy(20 + i * (card_width + 5), self.get_y())
            self.set_fill_color(*color)
            self.cell(card_width, 12, label, 0, 0, 'C', fill=True)
            self.set_xy(20 + i * (card_width + 5), self.get_y() + 12)
            self.set_fill_color(245, 245, 245)
            self.multi_cell(card_width, 6, value, 0, 'C', fill=True)
        self.ln(15)

    def add_forecast_charts(self, forecast_data):
        self._section_title("7-Day Forecast Trends")

        # AQI Forecast
        fig, ax = plt.subplots(figsize=(10, 4))
        dates = pd.to_datetime(forecast_data['date'])
        ax.plot(dates, forecast_data['aqi_prophet'], label='Prophet', color=MODEL_COLORS['prophet'])
        ax.plot(dates, forecast_data['aqi_xgb'], label='XGBoost', color=MODEL_COLORS['xgb'])
        for i in range(len(AQI_GRADIENT)):
            ax.axhspan(i * 50, (i + 1) * 50, facecolor=[x / 255 for x in AQI_GRADIENT[i]], alpha=0.2)
        ax.set_title("AQI Forecast Comparison", fontsize=12)
        ax.set_xlabel("Date")
        ax.set_ylabel("AQI")
        ax.legend()
        self._insert_plot(fig, "AQI Forecast")

        # Pollutant Forecasts
        for pollutant in POLLUTANTS:
            fig, ax = plt.subplots(figsize=(10, 3))
            ax.plot(dates, forecast_data[f'{pollutant}_prophet'], color=MODEL_COLORS['prophet'], label='Prophet')
            ax.plot(dates, forecast_data[f'{pollutant}_xgb'], color=MODEL_COLORS['xgb'], label='XGBoost')
            ax.set_title(f"{pollutant.upper()} Forecast", fontsize=10)
            ax.set_xlabel("Date")
            ax.set_ylabel("Concentration (μg/m³)")
            ax.tick_params(axis='both', which='major', labelsize=8)
            ax.legend()
            self._insert_plot(fig, f"{pollutant.upper()} Forecast")

    def _insert_plot(self, fig, title):
        tmp_file = f"{DATA_DIR}temp_plot.png"
        fig.savefig(tmp_file, bbox_inches='tight', dpi=120)
        self.image(tmp_file, x=20, w=self.w - 40)
        plt.close(fig)
        try:
            os.remove(tmp_file)
        except OSError as e:
            logging.warning(f"Failed to remove {tmp_file}: {e}")
        self.ln(5)

    def add_model_performance(self, metrics):
        self._section_title("Model Performance Metrics")
        col_widths = [40, 25, 25, 25, 25, 25, 25]
        self.set_font('Helvetica', 'B', 10)
        headers = ["Parameter", "P-MSE", "P-RMSE", "P-MAE", "X-MSE", "X-RMSE", "X-MAE"]
        for w, h in zip(col_widths, headers):
            self.cell(w, 8, h, 1, 0, 'C')
        self.ln()
        self.set_font('Helvetica', '', 9)
        for param, vals in metrics.items():
            self.cell(col_widths[0], 8, param.upper(), 1)
            self.cell(col_widths[1], 8, f"{vals['prophet_mse']:.2f}", 1)
            self.cell(col_widths[2], 8, f"{vals['prophet_rmse']:.2f}", 1)
            self.cell(col_widths[3], 8, f"{vals['prophet_mae']:.2f}", 1)
            self.cell(col_widths[4], 8, f"{vals['xgb_mse']:.2f}", 1)
            self.cell(col_widths[5], 8, f"{vals['xgb_rmse']:.2f}", 1)
            self.cell(col_widths[6], 8, f"{vals['xgb_mae']:.2f}", 1)
            self.ln()
        self.ln(10)

    def add_mortality_analysis(self, mortality_data, metrics, correlations):
        self._section_title("Mortality Analysis")
        if mortality_data is None:
            self.set_font('Helvetica', '', 10)
            self.set_text_color(255, 0, 0)
            self.cell(0, 10, "Mortality data unavailable or insufficient for analysis.", 0, 1, 'C')
            return

        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(mortality_data['date'], mortality_data['prophet'], label='Prophet', color=MODEL_COLORS['prophet'])
        ax.plot(mortality_data['date'], mortality_data['xgb'], label='XGBoost', color=MODEL_COLORS['xgb'])
        ax.fill_between(mortality_data['date'], mortality_data['prophet_lower'], mortality_data['prophet_upper'], color=MODEL_COLORS['prophet'], alpha=0.2)
        ax.set_title("30-Day Mortality Forecast", fontsize=12)
        ax.set_xlabel("Date")
        ax.set_ylabel("Deaths")
        ax.legend()
        self._insert_plot(fig, "Mortality Forecast")

        self.set_font('Helvetica', 'B', 10)
        self.cell(0, 8, "Pollutant-Death Correlations:", 0, 1)
        col_widths = [40, 30, 30]
        headers = ["Pollutant", "Correlation", "P-value"]
        for w, h in zip(col_widths, headers):
            self.cell(w, 8, h, 1, 0, 'C')
        self.ln()
        self.set_font('Helvetica', '', 9)
        for pollutant, (corr, pval) in correlations.items():
            self.cell(col_widths[0], 8, pollutant.upper(), 1)
            self.cell(col_widths[1], 8, f"{corr:.3f}" if not np.isnan(corr) else "N/A", 1)
            self.cell(col_widths[2], 8, f"{pval:.3f}" if not np.isnan(pval) else "N/A", 1)
            self.ln()

        self.ln(8)
        self.set_font('Helvetica', 'B', 10)
        self.cell(0, 8, "Mortality Model Metrics:", 0, 1)
        col_widths = [40, 30, 30, 30]
        headers = ["Model", "MSE", "RMSE", "MAE"]
        for w, h in zip(col_widths, headers):
            self.cell(w, 8, h, 1, 0, 'C')
        self.ln()
        self.set_font('Helvetica', '', 9)
        for model, vals in metrics.items():
            self.cell(col_widths[0], 8, model.capitalize(), 1)
            self.cell(col_widths[1], 8, f"{vals['mse']:.2f}", 1)
            self.cell(col_widths[2], 8, f"{vals['rmse']:.2f}", 1)
            self.cell(col_widths[3], 8, f"{vals['mae']:.2f}", 1)
            self.ln()

class AQIAnalyzer:
    """Class to analyze air quality and generate reports."""
    def __init__(self):
        self.models = {}
        self.metrics = {}
        self.mortality_metrics = {}
        self.historical_days = 365
        self.mortality_df = self.load_mortality_data()

    def load_mortality_data(self):
        mortality_file = f"{DATA_DIR}mortality_data.csv"
        if not os.path.exists(mortality_file):
            logging.error(f"Mortality data file not found: {mortality_file}")
            return None
        try:
            mortality = pd.read_csv(mortality_file, parse_dates=['DOD'], dayfirst=True)
            required_cols = ['DOD', 'MONITORING STATIONS', 'PM2.5', 'PM10', 'NO2', 'SO2', 'OZONE']
            if not all(col in mortality.columns for col in required_cols):
                missing = [col for col in required_cols if col not in mortality.columns]
                logging.error(f"Missing columns in mortality data: {missing}")
                return None
            mortality.rename(columns={'DOD': 'date', 'MONITORING STATIONS': 'Location'}, inplace=True)
            mortality['Location'] = mortality['Location'].apply(lambda x: x.split(',')[0])
            logging.info("Mortality data loaded successfully")
            return mortality
        except Exception as e:
            logging.error(f"Error loading mortality data: {str(e)}")
            return None

    def _safe_api_call(self, url, params, max_retries=3):
        for attempt in range(max_retries):
            try:
                response = requests.get(url, params=params, timeout=10)
                response.raise_for_status()
                return response.json()
            except Exception as e:
                logging.warning(f"API call failed (attempt {attempt + 1}/{max_retries}): {str(e)}")
                time.sleep(2)
        logging.error(f"API call failed after {max_retries} retries")
        return None

    def _fetch_data(self, lat, lon):
        url_aq = "https://air-quality-api.open-meteo.com/v1/air-quality"
        params_aq = {
            'latitude': lat,
            'longitude': lon,
            'hourly': ','.join(POLLUTANTS),
            'start_date': (datetime.now() - timedelta(days=self.historical_days)).strftime("%Y-%m-%d"),
            'end_date': datetime.now().strftime("%Y-%m-%d"),
            'timezone': 'auto'
        }
        data_aq = self._safe_api_call(url_aq, params_aq)
        if not data_aq or 'hourly' not in data_aq:
            logging.error(f"No air quality data retrieved for lat={lat}, lon={lon}")
            return pd.DataFrame()
        df_aq = pd.DataFrame({
            'date': pd.to_datetime(data_aq['hourly']['time']),
            **{p: data_aq['hourly'].get(p, [np.nan] * len(data_aq['hourly']['time'])) for p in POLLUTANTS}
        })
        df_aq = df_aq.set_index('date').resample('D').mean().reset_index()

        url_weather = "https://archive-api.open-meteo.com/v1/archive"
        params_weather = {
            'latitude': lat,
            'longitude': lon,
            'hourly': ','.join(['temperature_2m', 'relative_humidity_2m', 'wind_speed_10m']),
            'start_date': (datetime.now() - timedelta(days=self.historical_days)).strftime("%Y-%m-%d"),
            'end_date': datetime.now().strftime("%Y-%m-%d"),
            'timezone': 'auto'
        }
        data_weather = self._safe_api_call(url_weather, params_weather)
        if not data_weather or 'hourly' not in data_weather:
            logging.error(f"No weather data retrieved for lat={lat}, lon={lon}")
            return pd.DataFrame()
        df_weather = pd.DataFrame({
            'date': pd.to_datetime(data_weather['hourly']['time']),
            'temperature': data_weather['hourly'].get('temperature_2m', [np.nan] * len(data_weather['hourly']['time'])),
            'humidity': data_weather['hourly'].get('relative_humidity_2m', [np.nan] * len(data_weather['hourly']['time'])),
            'wind_speed': data_weather['hourly'].get('wind_speed_10m', [np.nan] * len(data_weather['hourly']['time']))
        })
        df_weather = df_weather.set_index('date').resample('D').mean().reset_index()

        merged = pd.merge(df_aq, df_weather, on='date', how='outer')
        merged = merged.dropna()
        if merged.empty:
            logging.warning(f"No valid data after merging for lat={lat}, lon={lon}")
        return merged

    def _train_pollutant_model(self, df, param):
        prophet_df = df[['date', param]].rename(columns={'date': 'ds', param: 'y'}).dropna()
        m, p_err, p_rmse, p_mae = None, np.inf, np.inf, np.inf
        if len(prophet_df) >= MIN_TRAINING_SAMPLES:
            try:
                m = Prophet(yearly_seasonality=True, weekly_seasonality=True)
                m.fit(prophet_df)
                preds = m.predict(prophet_df)['yhat']
                p_err = mean_squared_error(prophet_df['y'], preds)
                p_rmse = math.sqrt(p_err)
                p_mae = mean_absolute_error(prophet_df['y'], preds)
            except Exception as e:
                logging.error(f"Prophet failed for {param}: {str(e)}")

        x_err, x_rmse, x_mae, x_model = np.inf, np.inf, np.inf, None
        xdf = df.set_index('date').copy()
        for i in range(1, LAG_DAYS + 1):
            xdf[f'{param}_lag_{i}'] = xdf[param].shift(i)
        xdf.dropna(inplace=True)
        if len(xdf) >= MIN_TRAINING_SAMPLES:
            features = [f'{param}_lag_{i}' for i in range(1, LAG_DAYS + 1)] + WEATHER_PARAMS
            X, y = xdf[features], xdf[param]
            try:
                x_model = XGBRegressor(n_estimators=100, random_state=42)
                x_model.fit(X, y)
                preds = x_model.predict(X)
                x_err = mean_squared_error(y, preds)
                x_rmse = math.sqrt(x_err)
                x_mae = mean_absolute_error(y, preds)
            except Exception as e:
                logging.error(f"XGBoost failed for {param}: {str(e)}")

        return m, p_err, p_rmse, p_mae, x_model, x_err, x_rmse, x_mae

    def _generate_forecasts(self, df, station):
        dates = pd.date_range(start=datetime.now(), periods=7)
        res = pd.DataFrame({'date': dates.strftime('%Y-%m-%d')})
        for param in POLLUTANTS:
            res[f'{param}_prophet'] = np.nan
            res[f'{param}_xgb'] = np.nan
            if param in self.models[station]:
                info = self.models[station][param]
                prophet_fc = np.full(len(dates), np.nan)
                xgb_fc = np.full(len(dates), np.nan)
                if info['prophet']:
                    try:
                        prophet_fc = info['prophet'].predict(pd.DataFrame({'ds': dates}))['yhat'].values
                    except Exception as e:
                        logging.error(f"Prophet forecast failed for {param} in {station}: {str(e)}")
                if info['xgb']:
                    try:
                        window = list(df[param].values[-LAG_DAYS:])
                        xgb_fc = []
                        for _ in dates:
                            wf = [df[p].mean() for p in WEATHER_PARAMS if p in df]
                            xgb_fc.append(info['xgb'].predict([window[-LAG_DAYS:] + wf])[0])
                            window.append(xgb_fc[-1])
                        xgb_fc = np.array(xgb_fc)
                    except Exception as e:
                        logging.error(f"XGBoost forecast failed for {param} in {station}: {str(e)}")
                res[f'{param}_prophet'] = prophet_fc
                res[f'{param}_xgb'] = xgb_fc

        res['aqi_prophet'] = res.apply(lambda r: self._calculate_aqi_index(r, 'prophet'), axis=1)
        res['aqi_xgb'] = res.apply(lambda r: self._calculate_aqi_index(r, 'xgb'), axis=1)
        return res

    def _calculate_aqi_index(self, row, model_suffix):
        pollutants = {'pm2_5': 0.255, 'pm10': 0.2125, 'no2': 0.17, 'so2': 0.1275, 'o3': 0.085, 'co': 0.15}
        aqi = sum(row.get(f"{p}_{model_suffix}", 0) * w for p, w in pollutants.items())
        humidity_adj = row.get(f"humidity_{model_suffix}", 0) * 0.01
        wind_adj = row.get(f"wind_speed_{model_suffix}", 0) * 0.1
        temp_adj = row.get(f"temperature_{model_suffix}", 0) * 0.05
        return aqi * (1 + humidity_adj - wind_adj + temp_adj)

    def predict_mortality(self, station):
        if self.mortality_df is None:
            logging.warning(f"No mortality data available for {station}")
            return None, {}, {}

        station_data = self.mortality_df[self.mortality_df['Location'] == station].copy()
        if station_data.empty:
            logging.warning(f"No mortality data for {station}")
            return None, {}, {}

        daily_deaths = station_data.groupby('date').size().reset_index(name='deaths')
        daily_deaths.columns = ['ds', 'y']
        if len(daily_deaths) < MIN_TRAINING_SAMPLES:
            logging.warning(f"Insufficient mortality data for {station}: {len(daily_deaths)} samples")
            return None, {}, {}

        correlations = {}
        merged_data = station_data.groupby('date').agg({
            'pm25': 'mean', 'pm10': 'mean', 'no2': 'mean', 'so2': 'mean', 'o3': 'mean'
        }).reset_index()
        merged_data = merged_data.merge(daily_deaths, left_on='date', right_on='ds', how='inner')
        for pollutant in MORTALITY_PARAMS:
            if pollutant in merged_data and 'y' in merged_data:
                corr, pval = pearsonr(merged_data[pollutant].fillna(merged_data[pollutant].mean()), merged_data['y'])
                correlations[pollutant] = (corr, pval)
            else:
                correlations[pollutant] = (np.nan, np.nan)

        train = daily_deaths.iloc[:-30].copy()
        test = daily_deaths.iloc[-30:].copy()
        train = train.merge(merged_data[['ds'] + MORTALITY_PARAMS], on='ds', how='left')
        test = test.merge(merged_data[['ds'] + MORTALITY_PARAMS], on='ds', how='left')
        for pollutant in MORTALITY_PARAMS:
            train[pollutant] = train[pollutant].fillna(train[pollutant].mean())
            test[pollutant] = test[pollutant].fillna(test[pollutant].mean())

        prophet_metrics = {}
        prophet_pred = None
        try:
            prophet_model = Prophet(yearly_seasonality=True, weekly_seasonality=True)
            for pollutant in MORTALITY_PARAMS:
                corr = correlations.get(pollutant, (0, 1))[0]
                weight = abs(corr) if not np.isnan(corr) else 0
                prophet_model.add_regressor(pollutant, prior_scale=weight * 10)
            prophet_model.fit(train[['ds', 'y'] + MORTALITY_PARAMS])
            prophet_pred = prophet_model.predict(test[['ds'] + MORTALITY_PARAMS])
            prophet_metrics = {
                'mse': mean_squared_error(test['y'], prophet_pred['yhat']),
                'rmse': math.sqrt(mean_squared_error(test['y'], prophet_pred['yhat'])),
                'mae': mean_absolute_error(test['y'], prophet_pred['yhat'])
            }
        except Exception as e:
            logging.error(f"Prophet mortality model failed for {station}: {str(e)}")

        xgb_metrics = {}
        xgb_pred = None
        try:
            train['ordinal'] = train['ds'].map(datetime.toordinal)
            test['ordinal'] = test['ds'].map(datetime.toordinal)
            features = ['ordinal'] + MORTALITY_PARAMS
            for pollutant in MORTALITY_PARAMS:
                corr = correlations.get(pollutant, (0, 1))[0]
                weight = abs(corr) if not np.isnan(corr) else 0
                train[pollutant] *= weight
                test[pollutant] *= weight
            xgb_model = XGBRegressor(n_estimators=100, random_state=42)
            xgb_model.fit(train[features], train['y'])
            xgb_pred = pd.DataFrame({'ds': test['ds'], 'yhat': xgb_model.predict(test[features])})
            xgb_metrics = {
                'mse': mean_squared_error(test['y'], xgb_pred['yhat']),
                'rmse': math.sqrt(mean_squared_error(test['y'], xgb_pred['yhat'])),
                'mae': mean_absolute_error(test['y'], xgb_pred['yhat'])
            }
        except Exception as e:
            logging.error(f"XGBoost mortality model failed for {station}: {str(e)}")

        mortality_data = pd.DataFrame({
            'date': test['ds'],
            'prophet': prophet_pred['yhat'] if prophet_pred is not None else np.nan,
            'prophet_lower': prophet_pred['yhat_lower'] if prophet_pred is not None else np.nan,
            'prophet_upper': prophet_pred['yhat_upper'] if prophet_pred is not None else np.nan,
            'xgb': xgb_pred['yhat'] if xgb_pred is not None else np.nan
        })

        return mortality_data, {'prophet': prophet_metrics, 'xgb': xgb_metrics}, correlations

    def _health_risk_assessment(self, forecast):
        aqi_mean = forecast['aqi_prophet'].mean()
        if aqi_mean > 150:
            return "High Risk"
        elif aqi_mean > 100:
            return "Moderate Risk"
        else:
            return "Low Risk"

    def _risk_change_assessment(self, forecast):
        aqi_trend = forecast['aqi_prophet'].iloc[-1] - forecast['aqi_prophet'].iloc[0]
        if aqi_trend > 20:
            return "Increasing"
        elif aqi_trend < -20:
            return "Decreasing"
        else:
            return "Stable"

    def _get_aqi_status(self, aqi):
        for i, thresh in enumerate(AQI_THRESHOLDS):
            if aqi <= thresh:
                return AQI_STATUSES[i], i
        return AQI_STATUSES[-1], len(AQI_STATUSES) - 1

    def generate_report(self, station):
        try:
            lat, lon = STATIONS[station]
            df = self._fetch_data(lat, lon)
            if df.empty or len(df) < MIN_TRAINING_SAMPLES:
                logging.error(f"Insufficient data for {station}: {len(df)} samples")
                return

            self.models[station] = {}
            self.metrics[station] = {}
            for param in POLLUTANTS:
                pm, p_mse, p_rmse, p_mae, xm, x_mse, x_rmse, x_mae = self._train_pollutant_model(df, param)
                self.models[station][param] = {'prophet': pm, 'xgb': xm}
                self.metrics[station][param] = {
                    'prophet_mse': p_mse, 'prophet_rmse': p_rmse, 'prophet_mae': p_mae,
                    'xgb_mse': x_mse, 'xgb_rmse': x_rmse, 'xgb_mae': x_mae
                }

            forecast = self._generate_forecasts(df, station)
            mortality_data, mortality_metrics, correlations = self.predict_mortality(station)

            report = ModernReport(station)
            aqi_status, aqi_level = self._get_aqi_status(forecast['aqi_prophet'].iloc[0])
            dominant_idx = forecast[[f'{p}_prophet' for p in POLLUTANTS]].iloc[0].idxmax()
            dominant_pollutant = dominant_idx.replace('_prophet', '') if pd.notna(dominant_idx) else 'N/A'
            dominant_value = forecast[dominant_idx].iloc[0] if pd.notna(dominant_idx) else 0
            report.add_executive_summary({
                'current_aqi': forecast['aqi_prophet'].iloc[0],
                'aqi_status': aqi_status,
                'aqi_level': aqi_level,
                'dominant_pollutant': dominant_pollutant,
                'dominant_value': dominant_value,
                'risk_level': self._health_risk_assessment(forecast),
                'risk_change': self._risk_change_assessment(forecast)
            })

            report.add_forecast_charts(forecast)
            report.add_model_performance(self.metrics[station])
            report.add_mortality_analysis(mortality_data, mortality_metrics, correlations)
            report.output(f"{DATA_DIR}{station.replace(' ', '_')}_report.pdf")
            logging.info(f"Generated report for {station}")
        except Exception as e:
            logging.error(f"Failed to generate report for {station}: {str(e)}")

if __name__ == "__main__":
    analyzer = AQIAnalyzer()
    for station in STATIONS:
        analyzer.generate_report(station)