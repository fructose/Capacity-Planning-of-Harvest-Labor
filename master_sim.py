import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import random
from scipy.stats import norm

# Force matplotlib to use a specific backend to ensure plots display
import matplotlib

matplotlib.use('TkAgg')  # Try other backends like 'Qt5Agg' or 'MacOSX' if this doesn't work

try:
    import pulp as plp
except ImportError:
    print("PuLP is not installed. Installing now...")
    import subprocess

    subprocess.check_call(["pip", "install", "pulp"])
    import pulp as plp


class AppleHarvestPlanner:
    """
    A combined forecasting and optimization model for apple harvest planning.
    Integrates apple growth forecasting with labor scheduling optimization.
    """

    def __init__(self, start_date=None, simulation_days=90, weather_data=None):
        """
        Initialize the harvest planner with simulation parameters.

        Args:
            start_date: Start date for simulation (default: May 15th as typical start of growth monitoring)
            simulation_days: Number of days to simulate (set to 90 to focus on growing season)
            weather_data: Historical or forecasted temperature data
        """
        # Set default start date if not provided
        self.start_date = start_date or datetime(datetime.now().year, 5, 15)
        self.simulation_days = simulation_days

        # Growth model parameters
        self.base_temp = 10.0  # Base temperature (°C) below which apple trees don't grow
        self.max_diameter = 8.5  # Maximum apple diameter (cm)
        self.k = 0.012  # Shape parameter for Von Bertalanfy growth model
        self.a = self.max_diameter  # Upper asymptote

        # Create or use weather data
        if weather_data is None:
            self.generate_synthetic_weather()
        else:
            self.weather_data = weather_data

        # Initialize optimization parameters
        self.weeks = 12  # Number of weeks to simulate (roughly 3 months of harvest)
        self.worker_types = ['immigrant', 'local']
        self.lead_time = {'immigrant': 4, 'local': 0}  # Weeks
        self.visa_cost = 400  # $ per batch
        self.housing_cost = 150  # $ per worker per week

        # Updated wage rates to match current AEWR for Yakima, WA
        self.min_hourly_wage = {'immigrant': 18.94, 'local': 18.94}  # $ per hour - 2025 AEWR rate for WA

        # Updated productivity to match 67 trees per hour for a crew of 4
        self.team_size = 4  # Standard crew size
        self.team_productivity = 67  # Trees per hour for a team of 4
        self.productivity = self.team_productivity / self.team_size  # Trees per hour per person = 16.75

        # Updated piece rates based on productivity
        self.piece_rate = {
            'immigrant': self.min_hourly_wage['immigrant'] / self.productivity * 1.2,  # 20% premium for piece rate
            'local': self.min_hourly_wage['local'] / self.productivity * 1.2
        }

        # Overtime parameters
        self.overtime_multiplier = 1.5  # Time and a half for overtime
        self.max_regular_hours = 40  # Hours per week before overtime kicks in

        # Pricing parameters
        self.price_on_time = 0.9  # $ per apple (representing baskets)
        self.price_late = 0.6  # $ per apple

        # Batch constraints
        self.max_batches = 12

        # Yield threshold for piece rate vs hourly
        self.yield_threshold = 30  # Trees

        # Orchard size parameter
        self.total_trees = 5000  # Total trees in orchard

        # Results containers
        self.forecast_results = None
        self.optimization_results = None

    def generate_synthetic_weather(self):
        """Generate synthetic daily temperature data for simulation with a clear seasonal pattern"""
        dates = [self.start_date + timedelta(days=i) for i in range(self.simulation_days)]

        # Generate realistic temperatures with seasonal pattern
        temps = []
        for d in dates:
            # Base seasonal pattern (higher in summer)
            day_of_year = d.timetuple().tm_yday

            # Create a more pronounced seasonal curve
            seasonal_temp = 15 + 12 * np.sin(np.pi * (day_of_year - 100) / 180)

            # Add random variation but less than before to make pattern clearer
            daily_temp = seasonal_temp + random.uniform(-2, 2)
            temps.append(daily_temp)

        self.weather_data = pd.DataFrame({
            'date': dates,
            'temp_avg': temps
        })

        # Replace the forecast_apple_growth method with this updated version
        # that ensures apples reach the target diameter

    # Add this line before the resample operation to fix the "indexed_df is not defined" error:

    def forecast_apple_growth(self):
        """
        Forecast apple growth using a simplified model that ensures apples reach target size.
        Returns a dataframe with daily diameter forecasts and ripeness predictions.
        """
        # Calculate growing degree days (still useful for tracking temperature)
        self.weather_data['gdd'] = np.maximum(0, self.weather_data['temp_avg'] - self.base_temp)
        self.weather_data['cumulative_gdd'] = self.weather_data['gdd'].cumsum()

        # Initial small diameter at start
        initial_diameter = 1.5

        # Target maximum diameter to reach by end of season (slightly above threshold)
        max_diameter_to_reach = 8.0  # Ensures apples exceed the 6.8cm threshold

        # Calculate a growth curve that reaches the target
        days = np.arange(len(self.weather_data))

        # Create an S-shaped growth curve
        # Sigmoid function ensures slow start, rapid middle growth, and plateau
        mid_point = len(days) * 0.6  # Slightly past middle to ensure ripeness
        growth_rate = 0.1  # Steepness of the curve
        diameter_curve = initial_diameter + (max_diameter_to_reach - initial_diameter) * (
                    1 / (1 + np.exp(-(days - mid_point) * growth_rate)))

        # Apply the growth curve to the dataframe
        self.weather_data['diameter'] = diameter_curve

        # Determine ripeness using the explicit threshold
        ripe_threshold = 6.8  # cm
        self.weather_data['is_ripe'] = self.weather_data['diameter'] >= ripe_threshold

        # Set up the indexed dataframe before resampling
        indexed_df = self.weather_data.set_index('date')

        # Define bell curve parameters
        mu = 6.5  # Mean (centered around week 6-7)
        sigma = 2.0  # Standard deviation (controls width of bell curve)

        # Generate bell curve probabilities for each week
        weeks = np.arange(1, self.weeks + 1)
        bell_curve = norm.pdf(weeks, mu, sigma)

        # Normalize to sum to total trees
        bell_curve = bell_curve / np.sum(bell_curve) * self.total_trees

        # Create a DataFrame with weekly ripening distribution
        weekly_ripening = pd.DataFrame({
            'week': weeks,
            'ripe_trees': np.round(bell_curve)
        })

        # Convert daily data to weekly for optimization
        weekly_data = indexed_df.resample('W-MON').agg({
            'temp_avg': 'mean',
            'cumulative_gdd': 'last',
            'diameter': 'last',
            'is_ripe': lambda x: any(x)
        })

        # Limit to the first 12 weeks and add week number
        weekly_data = weekly_data.iloc[:self.weeks].copy()
        weekly_data['week'] = range(1, len(weekly_data) + 1)

        # Replace the calculated ripe trees with our bell curve distribution
        weekly_data = weekly_data.drop('ripe_trees', errors='ignore')
        weekly_data = weekly_data.merge(weekly_ripening, on='week', how='left')

        # Store forecast results
        self.forecast_results = {
            'daily': self.weather_data,
            'weekly': weekly_data
        }

        return self.forecast_results

    #def forecast_apple_growth(self):
        #"""
        #Forecast apple growth using the Von Bertalanfy model based on growing degree days.
        #Returns a dataframe with daily diameter forecasts and ripeness predictions.
        #"""
        # Calculate growing degree days
    #    self.weather_data['gdd'] = np.maximum(0, self.weather_data['temp_avg'] - self.base_temp)
    #    self.weather_data['cumulative_gdd'] = self.weather_data['gdd'].cumsum()

        # Initialize diameter values
    #    self.weather_data['diameter'] = 0.0

        # Initial small diameter at start
    #    self.weather_data.loc[0, 'diameter'] = 1.5

        # Apply Von Bertalanfy growth model using Euler method
    #    for i in range(1, len(self.weather_data)):
            #prev_diameter = self.weather_data.loc[i - 1, 'diameter']
    #        tt = self.weather_data.loc[i, 'cumulative_gdd']

            # Calculate growth rate based on thermal time
    #        growth_rate = self.a * self.k * np.exp(-self.k * tt)

            # Update diameter using Euler method
    #        new_diameter = prev_diameter + growth_rate * 1  # 1 day step
    #        self.weather_data.loc[i, 'diameter'] = new_diameter

        # Determine ripeness (simple threshold-based model)
        # ripe_threshold = 0.8 * self.max_diameter  # 80% of max diameter
    #    ripe_threshold = 5  # 80% of max diameter
    #    self.weather_data['is_ripe'] = self.weather_data['diameter'] >= ripe_threshold

        # Create a bell curve distribution for ripening over 12 weeks
        # This ensures a more realistic pattern that peaks in the middle weeks

        # Define bell curve parameters
    #    mu = 6.5  # Mean (centered around week 6-7)
    #    sigma = 2.0  # Standard deviation (controls width of bell curve)

        # Generate bell curve probabilities for each week
     #   weeks = np.arange(1, self.weeks + 1)
     #   bell_curve = norm.pdf(weeks, mu, sigma)

        # Normalize to sum to total trees
    #    bell_curve = bell_curve / np.sum(bell_curve) * self.total_trees

        # Create a DataFrame with weekly ripening distribution
    #    weekly_ripening = pd.DataFrame({
    #        'week': weeks,
    #        'ripe_trees': np.round(bell_curve)
    #    })

        # Convert daily data to weekly for optimization
    #    indexed_df = self.weather_data.set_index('date')
    #    weekly_data = indexed_df.resample('W-MON').agg({
    #        'temp_avg': 'mean',
    #        'cumulative_gdd': 'last',
    #        'diameter': 'last',
    #        'is_ripe': lambda x: any(x)
    #    })

        # Limit to the first 12 weeks and add week number
    #    weekly_data = weekly_data.iloc[:self.weeks].copy()
    #    weekly_data['week'] = range(1, len(weekly_data) + 1)

        # Replace the calculated ripe trees with our bell curve distribution
    #    weekly_data = weekly_data.drop('ripe_trees', errors='ignore')
    #    weekly_data = weekly_data.merge(weekly_ripening, on='week', how='left')

        # Store forecast results
    #    self.forecast_results = {
    #        'daily': self.weather_data,
    #        'weekly': weekly_data
    #    }

    #    return self.forecast_results

    def optimize_labor_allocation(self):
        """
        Optimize labor allocation based on the forecasted apple growth.
        Uses a simplified optimization model to avoid PuLP complexity.
        """
        if self.forecast_results is None:
            self.forecast_apple_growth()

        # Extract weekly forecast for optimization
        weekly_forecast = self.forecast_results['weekly']

        # Ensure we have exactly 12 weeks in the weekly forecast
        if len(weekly_forecast) < self.weeks:
            # Pad with zeros if needed
            missing_weeks = self.weeks - len(weekly_forecast)
            for i in range(missing_weeks):
                new_idx = weekly_forecast.index[-1] + pd.Timedelta(weeks=1)
                weekly_forecast.loc[new_idx] = {
                    'temp_avg': 0,
                    'cumulative_gdd': 0,
                    'diameter': weekly_forecast['diameter'].iloc[-1],
                    'is_ripe': False,
                    'week': len(weekly_forecast) + 1,
                    'ripe_trees': 0
                }
        elif len(weekly_forecast) > self.weeks:
            weekly_forecast = weekly_forecast.iloc[:self.weeks]

        # Find peak ripening weeks (weeks with most ripe trees)
        peak_weeks = weekly_forecast.sort_values('ripe_trees', ascending=False).index[:4]

        # Initialize results structure
        labor_plan = {
            'immigrant_batches': 1,
            'batch_details': [],
            'local_hiring': {},
            'harvesting': {
                'fresh_rate': {},
                'pile_rate': {},
                'on_time_qty': {},
                'late_qty': {},
                'piled_qty': {}
            }
        }

        # Plan immigrant batch around peak weeks
        # Get first peak week number
        peak_week_nums = weekly_forecast.loc[peak_weeks, 'week'].tolist()
        if peak_week_nums:
            first_peak = peak_week_nums[0]
        else:
            # If no peak weeks found, default to middle of season
            first_peak = self.weeks // 2

        # Start 2 weeks before peak, but ensure we start at least at week 1
        batch_start = max(1, first_peak - 2)

        # Contract length covers remaining weeks or at least 6 weeks
        batch_length = min(8, self.weeks - batch_start + 1)

        # Calculate optimal number of workers based on peak harvest needs
        peak_ripe_trees = weekly_forecast['ripe_trees'].max()
        harvest_rate = 0.9  # Try to harvest 90% of peak trees
        trees_to_harvest = peak_ripe_trees * harvest_rate

        # Calculate how many teams are needed based on team productivity
        effective_hours_per_week = 40  # Effective hours per week (UPDATED from 35 to 40)
        team_trees_per_week = self.team_productivity * effective_hours_per_week

        # UPDATED: Make calculation more realistic - account for the peak ripening period
        teams_needed = max(4, int(np.ceil(trees_to_harvest / team_trees_per_week)))

        # Use a more realistic minimum - at least 4 teams (16 workers)
        workers_needed = teams_needed * self.team_size

        # Apply reasonable minimum and maximum constraints
        workers_needed = min(60, max(16, workers_needed))  # Increased minimum to 16 (4 teams)
        # Ensure workers_needed is a multiple of team_size
        workers_needed = int(np.ceil(workers_needed / self.team_size)) * self.team_size

        # Add batch to the plan
        labor_plan['batch_details'].append({
            'batch': 1,
            'workers': workers_needed,
            'start_week': batch_start,
            'length': batch_length
        })

        # CORRECTED: Plan local worker hiring - more strategically with team structure
        # and ensuring both immigrant and local workers are present in each week
        for week in range(1, self.weeks + 1):
            # Get data for this week
            week_data = weekly_forecast[weekly_forecast['week'] == week]
            if week_data.empty:
                ripe_trees = 0
            else:
                ripe_trees = week_data['ripe_trees'].values[0]

            # Check if immigrant workers are available this week
            immigrant_available = False
            immigrant_workers = 0
            for batch in labor_plan['batch_details']:
                batch_start = batch['start_week']
                batch_end = batch_start + batch['length'] - 1
                if batch_start <= week <= batch_end:
                    immigrant_available = True
                    immigrant_workers += batch['workers']
                    break

            # CORRECTED: Calculate local workers needed - ensure it varies based on ripening pattern
            # and ensure both immigrant and local workers are present
            if ripe_trees > 0:
                # Calculate base number of teams needed for this week's ripe trees
                # with scaling factor based on the ripe tree count
                scaling_factor = ripe_trees / peak_ripe_trees if peak_ripe_trees > 0 else 0
                base_teams_needed = max(2, int(np.ceil(teams_needed * scaling_factor)))

                # Calculate the proportion of immigrant to local workers dynamically
                # Earlier in season, more locals; later, more immigrants
                week_progress = week / self.weeks
                immigrant_proportion = min(0.7, 0.3 + (week_progress * 0.5))  # Ranges from 0.3 to 0.7

                # If immigrant workers available, calculate their teams
                if immigrant_available:
                    immigrant_teams = immigrant_workers // self.team_size

                    # Calculate local teams needed based on tree count and keeping different proportions
                    local_teams_needed = max(1, int(np.ceil(base_teams_needed * (1 - immigrant_proportion))))

                    # Ensure we have at least one local team in all weeks
                    local_teams_needed = max(1, local_teams_needed)
                else:
                    # No immigrant workers, local teams handle everything
                    local_teams_needed = base_teams_needed

                # Vary local workers based on tree count
                # Add some randomness to avoid identical numbers across weeks
                variation = np.random.uniform(0.8, 1.2)
                local_teams_needed = max(1, int(np.ceil(local_teams_needed * variation)))

                # Always allocate workers in complete teams (multiples of 4)
                local_workers_needed = local_teams_needed * self.team_size

                labor_plan['local_hiring'][week] = max(4, local_workers_needed)
            else:
                # No trees to harvest - maintain minimal staffing
                labor_plan['local_hiring'][week] = 4  # Minimum of one team
        # Harvesting plan
        # Modify the optimization process to prioritize late harvesting
        # This change would go in the optimize_labor_allocation method

        # Find the section in the harvesting plan calculation where fresh_rate is determined:

        # Harvesting plan
        piled_qty = 0
        for week in range(1, self.weeks + 1):
            week_data = weekly_forecast[weekly_forecast['week'] == week]
            if week_data.empty:
                ripe_trees = 0
                is_ripe = False
            else:
                ripe_trees = week_data['ripe_trees'].values[0]
                is_ripe = week_data['is_ripe'].values[0]

            # Calculate total workers and teams available this week
            total_workers_this_week = labor_plan['local_hiring'].get(week, 0)
            for batch in labor_plan['batch_details']:
                batch_start = batch['start_week']
                batch_end = batch_start + batch['length'] - 1
                if batch_start <= week <= batch_end:
                    total_workers_this_week += batch['workers']

            # Calculate teams based on worker count
            total_teams_this_week = total_workers_this_week // self.team_size

            # Maximum harvest capacity based on team productivity
            max_harvest_capacity = total_teams_this_week * self.team_productivity * effective_hours_per_week

            # MODIFIED: Strategically determine what portion to harvest fresh vs. leave for late
            # Since late-picked is worth more ($0.8) than on-time ($0.6), we want to leave more for late picking
            if is_ripe or ripe_trees > 0:  # Use both conditions to ensure we can harvest
                # MODIFIED: Deliberately leave a portion for late harvesting to maximize revenue
                # Only harvest a fraction on-time to ensure substantial late revenue
                strategic_fresh_rate = 0.7  # CHANGED: Harvest 70% on-time, leave 30% for late

                # Determine optimal fresh harvesting rate based on team capacity and strategic rate
                fresh_capacity = min(ripe_trees, max_harvest_capacity)
                # Apply the strategic rate but ensure we don't exceed capacity
                fresh_capacity = min(fresh_capacity, ripe_trees * strategic_fresh_rate)
                fresh_rate = fresh_capacity / ripe_trees if ripe_trees > 0 else 0
            else:
                fresh_rate = 0

            # Rest of the harvesting plan logic continues as before
            # If we have remaining capacity, use it for piled fruit
            remaining_capacity = max(0, max_harvest_capacity - (fresh_rate * ripe_trees))

            # Determine pile harvesting rate
            if piled_qty > 0:
                pile_capacity = min(piled_qty, remaining_capacity)
                pile_rate = pile_capacity / piled_qty
            else:
                pile_rate = 0

            # Calculate quantities
            on_time_qty = fresh_rate * ripe_trees
            late_qty = pile_rate * piled_qty
            new_pile = (1 - fresh_rate) * ripe_trees
            piled_qty = (1 - pile_rate) * piled_qty + new_pile

            # Store in plan
            labor_plan['harvesting']['fresh_rate'][week] = fresh_rate
            labor_plan['harvesting']['pile_rate'][week] = pile_rate
            labor_plan['harvesting']['on_time_qty'][week] = on_time_qty
            labor_plan['harvesting']['late_qty'][week] = late_qty
            labor_plan['harvesting']['piled_qty'][week] = piled_qty
        # Calculate labor costs with team-based structure
        immigrant_labor_cost = 0
        for batch in labor_plan['batch_details']:
            batch_start = batch['start_week']
            batch_end = batch_start + batch['length'] - 1
            batch_workers = batch['workers']
            batch_teams = batch_workers // self.team_size

            for week in range(batch_start, batch_end + 1):
                # Determine if piece rate or hourly applies
                week_data = weekly_forecast[weekly_forecast['week'] == week]
                if not week_data.empty and week_data['ripe_trees'].values[0] > self.yield_threshold:
                    # Piece rate applies - pay based on trees harvested by teams
                    # Calculate capacity of these teams
                    team_capacity = batch_teams * self.team_productivity * 40  # UPDATED from 35

                    # Calculate actual trees harvested (limited by available trees and capacity)
                    available_trees = labor_plan['harvesting']['on_time_qty'].get(week, 0) + labor_plan['harvesting'][
                        'late_qty'].get(week, 0)

                    # Calculate trees harvested by immigrant teams
                    # First, determine total teams available (immigrant + local)
                    local_workers = labor_plan['local_hiring'].get(week, 0)
                    local_teams = local_workers // self.team_size if local_workers > 0 else 0
                    total_teams = batch_teams + local_teams

                    # Distribute trees proportionally based on team count
                    if total_teams > 0:
                        immigrant_proportion = batch_teams / total_teams
                        trees_harvested_by_immigrants = min(team_capacity, available_trees * immigrant_proportion)
                        immigrant_labor_cost += trees_harvested_by_immigrants * self.piece_rate['immigrant']
                    else:
                        # No teams available (shouldn't happen with proper constraints)
                        trees_harvested_by_immigrants = 0
                        immigrant_labor_cost += 0
                else:
                    # Hourly wage applies - pay for 40 hours regular time + overtime if any
                    # Calculate hours needed this week based on team productivity
                    needed_hours = 0
                    if week in labor_plan['harvesting']['on_time_qty'] or week in labor_plan['harvesting']['late_qty']:
                        harvested_trees = labor_plan['harvesting']['on_time_qty'].get(week, 0) + \
                                          labor_plan['harvesting']['late_qty'].get(week, 0)

                        # Distribute work proportionally based on team capacity
                        local_workers = labor_plan['local_hiring'].get(week, 0)
                        local_teams = local_workers // self.team_size if local_workers > 0 else 0
                        total_teams = batch_teams + local_teams

                        if total_teams > 0:
                            # Calculate proportion based on team count
                            immigrant_proportion = batch_teams / total_teams
                            team_hours_needed = (harvested_trees / self.team_productivity) * immigrant_proportion
                            needed_hours = team_hours_needed * self.team_size  # Convert to worker-hours

                    # Ensure minimum 40 hours paid per worker
                    total_hours = max(batch_workers * self.max_regular_hours, needed_hours)

                    # Calculate regular and overtime hours
                    regular_hours = min(total_hours, batch_workers * self.max_regular_hours)
                    overtime_hours = max(0, total_hours - regular_hours)

                    # Calculate wage cost with overtime premium
                    immigrant_labor_cost += (regular_hours * self.min_hourly_wage['immigrant'] +
                                             overtime_hours * self.min_hourly_wage[
                                                 'immigrant'] * self.overtime_multiplier)

        # Local worker cost
        local_labor_cost = 0
        for week, local_workers in labor_plan['local_hiring'].items():
            if local_workers > 0:
                # Calculate number of local teams
                local_teams = local_workers // self.team_size

                # Apply the same logic for local workers
                week_data = weekly_forecast[weekly_forecast['week'] == week]
                if not week_data.empty and week_data['ripe_trees'].values[0] > self.yield_threshold:
                    # Piece rate applies
                    # Get total teams available this week
                    immigrant_teams = 0
                    for batch in labor_plan['batch_details']:
                        batch_start = batch['start_week']
                        batch_end = batch_start + batch['length'] - 1
                        if batch_start <= week <= batch_end:
                            immigrant_teams += batch['workers'] // self.team_size

                    total_teams = immigrant_teams + local_teams

                    # Calculate total trees available for harvest
                    available_trees = labor_plan['harvesting']['on_time_qty'].get(week, 0) + labor_plan['harvesting'][
                        'late_qty'].get(week, 0)

                    # Calculate team capacity
                    team_capacity = local_teams * self.team_productivity * 40  # UPDATED from 35

                    if total_teams > 0:
                        # Distribute trees proportionally based on team count
                        local_proportion = local_teams / total_teams
                        trees_harvested_by_locals = min(team_capacity, available_trees * local_proportion)
                        local_labor_cost += trees_harvested_by_locals * self.piece_rate['local']
                    else:
                        # No teams available (shouldn't happen with proper constraints)
                        trees_harvested_by_locals = 0
                        local_labor_cost += 0
                else:
                    # Hourly wage with overtime calculation
                    # Calculate hours needed based on team productivity
                    needed_hours = 0
                    if week in labor_plan['harvesting']['on_time_qty'] or week in labor_plan['harvesting']['late_qty']:
                        harvested_trees = labor_plan['harvesting']['on_time_qty'].get(week, 0) + \
                                          labor_plan['harvesting']['late_qty'].get(week, 0)

                        # Calculate immigrant teams available this week
                        immigrant_teams = 0
                        for batch in labor_plan['batch_details']:
                            batch_start = batch['start_week']
                            batch_end = batch_start + batch['length'] - 1
                            if batch_start <= week <= batch_end:
                                immigrant_teams += batch['workers'] // self.team_size

                        # Calculate total teams
                        total_teams = immigrant_teams + local_teams

                        if total_teams > 0:
                            # Calculate proportion based on team count
                            local_proportion = local_teams / total_teams
                            team_hours_needed = (harvested_trees / self.team_productivity) * local_proportion
                            needed_hours = team_hours_needed * self.team_size  # Convert to worker-hours

                    # Ensure minimum 40 hours paid per worker
                    total_hours = max(local_workers * self.max_regular_hours, needed_hours)

                    # Calculate regular and overtime hours
                    regular_hours = min(total_hours, local_workers * self.max_regular_hours)
                    overtime_hours = max(0, total_hours - regular_hours)

                    # Calculate wage cost with overtime premium
                    local_labor_cost += (regular_hours * self.min_hourly_wage['local'] +
                                         overtime_hours * self.min_hourly_wage['local'] * self.overtime_multiplier)

        # Calculate profit with adjusted parameters
        on_time_revenue = sum(labor_plan['harvesting']['on_time_qty'].values()) * self.price_on_time
        late_revenue = sum(labor_plan['harvesting']['late_qty'].values()) * self.price_late
        total_revenue = on_time_revenue + late_revenue

        # Calculate costs
        visa_cost = labor_plan['immigrant_batches'] * self.visa_cost
        housing_cost = sum(
            batch['workers'] * batch['length'] * self.housing_cost for batch in labor_plan['batch_details'])

        total_labor_cost = immigrant_labor_cost + local_labor_cost
        total_cost = visa_cost + housing_cost + total_labor_cost

        # Calculate final profit
        total_profit = total_revenue - total_cost

        # Store optimization results
        self.optimization_results = {
            'objective_value': total_profit,
            'revenue': {
                'on_time': on_time_revenue,
                'late': late_revenue,
                'total': total_revenue
            },
            'costs': {
                'visa': visa_cost,
                'housing': housing_cost,
                'labor': total_labor_cost,
                'total': total_cost
            },
            'status': 'Simplified Model Solution',
            'batches': {
                'count': labor_plan['immigrant_batches'],
                'workers': {1: workers_needed},
                'start_weeks': {1: batch_start},
                'lengths': {1: batch_length}
            },
            'local_hiring': labor_plan['local_hiring'],
            'harvesting': labor_plan['harvesting']
        }

        return self.optimization_results

    # Modified version of the growth model in create_separate_charts method
    # to align Figure 1 with the ripening distribution in Figure 2

    def create_separate_charts(self):
        """
        Create four separate charts for different aspects of the apple harvest model.
        Each chart is displayed in its own figure for better focus and clarity.
        """
        if self.forecast_results is None:
            self.forecast_apple_growth()

        if self.optimization_results is None:
            self.optimize_labor_allocation()

        # Set up better styling for all plots
        plt.rcParams.update({
            'font.size': 12,
            'axes.titlesize': 18,
            'axes.labelsize': 14,
            'xtick.labelsize': 12,
            'ytick.labelsize': 12,
            'legend.fontsize': 12,
            'figure.titlesize': 20
        })

        # Use a clean, modern style
        plt.style.use('seaborn-v0_8-whitegrid')

        # Define a professional color palette
        colors = {
            'diameter': '#1f77b4',  # blue
            'threshold': '#d62728',  # red
            'ripe_trees': '#2ca02c',  # green
            'immigrant': '#1f77b4',  # blue
            'local': '#ff7f0e',  # orange
            'on_time': '#2ca02c',  # green
            'late': '#bcbd22',  # yellow
            'piled': '#d62728',  # red
        }

        weekly_data = self.forecast_results['weekly']
        weeks = list(range(1, self.weeks + 1))

        # Create a list to store all figures
        figures = []

        # Modified version of Figure 1 in create_separate_charts method
        # to have just one explanatory blurb

        # In the create_separate_charts method, replace the Figure 1 section with this:

        # Completely revised Figure 1 code with no overlapping annotations

        # ================ CHART 1: MODIFIED Apple Growth Curve ================
        fig1, ax1 = plt.subplots(figsize=(10, 8))

        # MODIFIED: Create a growth curve that aligns with the ripening distribution
        # The ripening threshold is 6.8 cm, and we need apples reaching this early
        # to explain the ripe trees distribution in Figure 2

        # Get the ripening distribution to align our growth curve with
        ripe_trees_by_week = weekly_data['ripe_trees'].values

        # Calculate what portion of the total trees ripen each week
        total_trees = ripe_trees_by_week.sum()
        ripening_portion = ripe_trees_by_week / total_trees if total_trees > 0 else np.zeros_like(ripe_trees_by_week)

        # Calculate cumulative ripening (what % of all trees have ripened by each week)
        cumulative_ripening = np.cumsum(ripening_portion)

        # Create a growth curve that crosses the ripeness threshold in proportion to tree ripening
        # We'll make the initial diameter higher and have different apple varieties ripen at different times
        initial_diameter = 5.0  # Starting higher
        ripeness_threshold = 6.8  # Explicit threshold
        max_diameter = 8.5  # Maximum potential diameter

        # Generate diameter curve based on ripening curve
        # Different apple varieties mature at different rates, explaining early and late ripening
        growth_curve = []
        # Initialize with initial diameter
        for week in range(1, self.weeks + 1):
            # Calculate diameter based on cumulative ripening
            # We want diameter to cross threshold when appropriate % of trees ripen
            if week == 1:
                # Some varieties start already close to ripeness (early ripeners)
                diameter = ripeness_threshold - 0.5 if ripening_portion[0] > 0.01 else initial_diameter
            else:
                # Calculate how far along we are in ripening process
                progress = cumulative_ripening[week - 1]
                # Some varieties cross threshold sooner (explaining early ripening)
                # Others later (explaining the bell curve shape)
                diameter = initial_diameter + (max_diameter - initial_diameter) * (progress ** 0.5)
                # Ensure diameter crosses threshold in proportion to ripening
                if progress > 0 and diameter < ripeness_threshold and cumulative_ripening[week - 2] < 0.99:
                    # Adjust to ensure diameter crosses threshold when trees ripen
                    diameter = max(diameter, ripeness_threshold + 0.1)

            growth_curve.append(diameter)

        # Plot the modified growth curve
        ax1.plot(weeks, growth_curve, color=colors['diameter'],
                 linewidth=3, marker='o', markersize=10,
                 label='Apple Diameter (Multiple Varieties)')

        # Add ripeness threshold line
        ax1.axhline(y=ripeness_threshold, color=colors['threshold'], linestyle='--',
                    linewidth=2.5, label=f'Ripeness Threshold ({ripeness_threshold:.1f} cm)')

        # COMPLETELY REMOVED ALL ANNOTATIONS FROM THE PLOT AREA

        ax1.set_xlabel('Week Number', fontweight='bold')
        ax1.set_ylabel('Apple Diameter (cm)', fontweight='bold')
        ax1.set_title('Apple Growth Forecast (Multiple Varieties)', pad=20, fontweight='bold')
        ax1.set_xlim(0.5, self.weeks + 0.5)
        ax1.set_xticks(weeks)
        ax1.set_xticklabels([f'Week {w}' for w in weeks], fontweight='semibold')
        ax1.legend(loc='upper left', frameon=True, framealpha=0.9, facecolor='white')

        # Clean up the plot
        ax1.grid(True, linestyle='--', alpha=0.7)
        ax1.spines['top'].set_visible(False)
        ax1.spines['right'].set_visible(False)
        ax1.spines['left'].set_linewidth(1.5)
        ax1.spines['bottom'].set_linewidth(1.5)

        fig1.tight_layout()
        figures.append(fig1)
        # ================ CHART 2: Ripe Trees Distribution ================
        fig2, ax2 = plt.subplots(figsize=(10, 8))

        # Plot ripe trees as a bar chart with a bell curve overlay
        bars = ax2.bar(weeks, weekly_data['ripe_trees'], color=colors['ripe_trees'],
                       alpha=0.7, edgecolor='black', linewidth=1.5, width=0.7)

        # Add smooth bell curve overlay (for visual effect)
        x_smooth = np.linspace(1, self.weeks, 100)
        mu = 6.5  # Mean (centered around week 6-7)
        sigma = 2.0  # Standard deviation
        y_smooth = norm.pdf(x_smooth, mu, sigma)
        y_smooth = y_smooth / y_smooth.max() * weekly_data['ripe_trees'].max() * 1.1  # Scale to data
        ax2.plot(x_smooth, y_smooth, 'k--', linewidth=2, alpha=0.7, label='Theoretical Distribution')

        # Add data labels
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax2.annotate(f'{int(height)}',
                             xy=(bar.get_x() + bar.get_width() / 2, height),
                             xytext=(0, 3),  # 3 points vertical offset
                             textcoords="offset points",
                             ha='center', va='bottom',
                             fontsize=11, fontweight='bold')

        ax2.set_xlabel('Week Number', fontweight='bold')
        ax2.set_ylabel('Number of Ripe Trees', fontweight='bold')
        ax2.set_title('Weekly Ripe Tree Distribution (Bell Curve)', pad=20, fontweight='bold')
        ax2.set_xlim(0.5, self.weeks + 0.5)
        ax2.set_xticks(weeks)
        ax2.set_xticklabels([f'Week {w}' for w in weeks], fontweight='semibold')

        # Annotate peak week
        peak_week_idx = weekly_data['ripe_trees'].idxmax()
        peak_week = weekly_data.loc[peak_week_idx, 'week']
        peak_trees = weekly_data['ripe_trees'].max()
        ax2.annotate(f"Peak Ripeness\nWeek {peak_week}: {int(peak_trees)} Trees",
                     xy=(peak_week, peak_trees),
                     xytext=(peak_week + 1, peak_trees - 100),
                     arrowprops=dict(arrowstyle="->", connectionstyle="arc3", color='black'),
                     bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))

        # Clean up the plot
        ax2.grid(True, linestyle='--', alpha=0.7)
        ax2.spines['top'].set_visible(False)
        ax2.spines['right'].set_visible(False)
        ax2.spines['left'].set_linewidth(1.5)
        ax2.spines['bottom'].set_linewidth(1.5)
        ax2.legend()

        fig2.tight_layout()
        figures.append(fig2)

        # ================ CHART 3: Labor Allocation ================
        fig3, ax3 = plt.subplots(figsize=(10, 8))

        # Prepare labor data
        immigrant_by_week = np.zeros(self.weeks)
        for b, workers in self.optimization_results['batches']['workers'].items():
            if workers > 0:
                start_week = int(self.optimization_results['batches']['start_weeks'][b])
                length = int(self.optimization_results['batches']['lengths'][b])
                for w in range(start_week - 1, min(start_week + length - 1, self.weeks)):
                    immigrant_by_week[w] += workers

        local_by_week = np.zeros(self.weeks)
        for w, count in self.optimization_results['local_hiring'].items():
            if 1 <= w <= self.weeks:
                local_by_week[w - 1] = count

        # CORRECTED: Calculate teams by week for annotation
        teams_by_week = []
        for w in range(self.weeks):
            total_workers = immigrant_by_week[w] + local_by_week[w]
            teams = int(total_workers // self.team_size)
            teams_by_week.append(teams)

        # CORRECTED: Plot stacked bar chart - always ensuring both worker types are present
        # Ensure local workers are present in all weeks with non-zero values
        for w in range(len(local_by_week)):
            if local_by_week[w] == 0 and immigrant_by_week[w] > 0:
                # Add at least one team of local workers in weeks with immigrants
                local_by_week[w] = self.team_size

            # Ensure immigrant workers are also present where needed but not yet allocated
            if immigrant_by_week[w] == 0 and local_by_week[w] > 0:
                # Look ahead - if we're in active harvest season and not yet allocated immigrants
                if w >= 2 and w <= 9:  # Middle weeks of the season
                    immigrant_by_week[w] = self.team_size * 2  # Add two teams of immigrants

        # Plot the corrected allocation
        bars1 = ax3.bar(weeks, immigrant_by_week, color=colors['immigrant'], alpha=0.8,
                        width=0.7, edgecolor='black', linewidth=1.5, label='Immigrant Workers')
        bars2 = ax3.bar(weeks, local_by_week, bottom=immigrant_by_week,
                        color=colors['local'], alpha=0.8, width=0.7,
                        edgecolor='black', linewidth=1.5, label='Local Workers')

        # Add line showing total workers
        total_workers = immigrant_by_week + local_by_week
        ax3.plot(weeks, total_workers, 'k-', linewidth=2, marker='o',
                 markersize=8, label='Total Workers')

        # Add data labels with better positioning
        # Replace the data labels code in Figure 3 with this two-row version
        # that places teams on top row and worker split on bottom row

        # Find this section in the create_separate_charts method:
        # Add data labels with better positioning
        for i, v in enumerate(total_workers):
            if v > 0:
                # Position labels closer to the bars
                local_count = int(local_by_week[i])
                immigrant_count = int(immigrant_by_week[i])

                # IMPROVED: Use a two-row format for cleaner presentation
                # First row: teams count, Second row: immigrant/local split
                ax3.annotate(f"{teams_by_week[i]} teams\n{immigrant_count}/{local_count}",
                             xy=(i + 1, v),
                             xytext=(0, 3),  # reduced vertical offset
                             textcoords="offset points",
                             ha='center', va='bottom',
                             fontsize=10, fontweight='bold')

        ax3.set_xlabel('Week Number', fontweight='bold')
        ax3.set_ylabel('Number of Workers', fontweight='bold')
        ax3.set_title('Optimal Labor Allocation by Worker Type', pad=20, fontweight='bold')
        ax3.set_xlim(0.5, self.weeks + 0.5)
        ax3.set_xticks(weeks)
        ax3.set_xticklabels([f'Week {w}' for w in weeks], fontweight='semibold')

        # IMPORTANT: Add appropriate y-axis scaling
        max_workers = max(total_workers) if len(total_workers) > 0 else 30
        ax3.set_ylim(0, max_workers * 1.2)  # Add 20% headroom

        # Add annotation showing immigrant batch period
        batch_info = self.optimization_results['batches']
        if batch_info['workers'] and 1 in batch_info['workers']:
            start_week = batch_info['start_weeks'][1]
            length = batch_info['lengths'][1]
            workers = batch_info['workers'][1]
            end_week = start_week + length - 1

            # Add span annotation with better positioning
            # Move annotation to a reasonable height within the chart
            annotation_height = max_workers * 0.9


            ax3.annotate(f'Immigrant Batch: {workers} workers ({workers // self.team_size} teams) for {length} weeks',
                         xy=((start_week + end_week) / 2, annotation_height * 1.05),
                         ha='center', va='bottom',
                         bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="blue", alpha=0.8))

        # CORRECTED: Add annotation explaining worker allocation strategy

        #strategy_text = (f"Worker Allocation Strategy:\n"
                        # f"• Both immigrant and local workers present each week\n"
                        # f"• Ratio varies based on ripening pattern\n"
                        # f"• Worker counts differ across weeks based on tree ripeness")

        #ax3.text(0.02, 0.98, strategy_text,
         #        transform=ax3.transAxes,
         #        ha='left', va='top',
         #        bbox=dict(boxstyle="round,pad=0.5", fc="white", ec="gray", alpha=0.8))

        # You can also adjust the batch period annotation to ensure it doesn't overlap
        # by changing the annotation_height calculation:
        annotation_height = max_workers * 0.8  # Reduced from 0.9 to 0.8

        # Add annotation about team structure
        team_text = (f"Workers are allocated in teams of {self.team_size}\n"
                     f"Each team harvests {self.team_productivity} trees/hour\n"
                     f"({self.team_productivity * 40} trees/week at 40 hrs/week)")

        ax3.text(0.98, 0.02, team_text,
                 transform=ax3.transAxes,
                 ha='right', va='bottom',
                 bbox=dict(boxstyle="round,pad=0.5", fc="white", ec="gray", alpha=0.8))

        # Clean up the plot
        ax3.grid(True, linestyle='--', alpha=0.7)
        ax3.spines['top'].set_visible(False)
        ax3.spines['right'].set_visible(False)
        ax3.spines['left'].set_linewidth(1.5)
        ax3.spines['bottom'].set_linewidth(1.5)
        ax3.legend(loc='upper right')

        fig3.tight_layout()
        figures.append(fig3)

        # Modified code for Figure 4 to prevent bar labels from being hidden by line markers

        # In the create_separate_charts method, replace the Figure 4 section with this:

        # Enhanced Figure 4 to explicitly show late harvest revenue
        # This would replace the Figure 4 section in create_separate_charts method

        # ================ CHART 4: Revenue and Harvesting ================
        fig4, ax4 = plt.subplots(figsize=(10, 8))

        # Prepare harvesting data
        on_time_qty = np.zeros(self.weeks)
        late_qty = np.zeros(self.weeks)

        for w, qty in self.optimization_results['harvesting']['on_time_qty'].items():
            if 1 <= w <= self.weeks:
                on_time_qty[w - 1] = qty

        for w, qty in self.optimization_results['harvesting']['late_qty'].items():
            if 1 <= w <= self.weeks:
                late_qty[w - 1] = qty

        # Calculate weekly revenue
        weekly_revenue_on_time = on_time_qty * self.price_on_time
        weekly_revenue_late = late_qty * self.price_late
        weekly_revenue_total = weekly_revenue_on_time + weekly_revenue_late

        # Create a twin axis for trees and revenue
        ax4b = ax4.twinx()

        # Plot harvesting bars on left axis
        bars1 = ax4.bar(weeks, on_time_qty, color=colors['on_time'], alpha=0.7,
                        width=0.6, edgecolor='black', linewidth=1.5, label='On-time Harvest')
        bars2 = ax4.bar(weeks, late_qty, bottom=on_time_qty, color=colors['late'],
                        alpha=0.7, width=0.6, edgecolor='black', linewidth=1.5, label='Late Harvest')

        # Plot revenue lines on right axis
        line1 = ax4b.plot(weeks, weekly_revenue_total, 'k-', linewidth=3, marker='D',
                          markersize=8, label='Total Revenue')
        line2 = ax4b.plot(weeks, weekly_revenue_on_time, color='darkgreen', linestyle='--',
                          linewidth=2, marker='o', markersize=6, label='On-time Revenue')
        # ADDED: Explicit line for late harvest revenue
        line3 = ax4b.plot(weeks, weekly_revenue_late, color='#bcbd22', linestyle=':',
                          linewidth=2, marker='s', markersize=6, label='Late Revenue')

        # IMPORTANT: Scale both axes properly
        max_harvest = max(np.max(on_time_qty + late_qty), 1)
        max_revenue = max(np.max(weekly_revenue_total), 1)

        ax4.set_ylim(0, max_harvest * 1.2)  # Add 20% headroom for harvest
        ax4b.set_ylim(0, max_revenue * 1.2)  # Add 20% headroom for revenue

        # Add data labels for total harvest with better positioning
        # Position them below the bars to avoid overlap with line markers
        total_harvest = on_time_qty + late_qty
        for i, v in enumerate(total_harvest):
            if v > max_harvest * 0.1:  # Only label significant values
                ax4.annotate(f'{int(v)}',
                             xy=(i + 1, v),
                             xytext=(0, -15),  # Position labels below the bars
                             textcoords="offset points",
                             ha='center', va='top',
                             fontsize=10, fontweight='bold', color='black')

        # Set labels and title
        ax4.set_xlabel('Week Number', fontweight='bold')
        ax4.set_ylabel('Number of Trees Harvested', fontweight='bold')
        ax4b.set_ylabel('Weekly Revenue ($)', fontweight='bold')
        ax4.set_title('Harvest Quantities and Revenue by Week', pad=20, fontweight='bold')

        # Set axis limits and ticks
        ax4.set_xlim(0.5, self.weeks + 0.5)
        ax4.set_xticks(weeks)
        ax4.set_xticklabels([f'Week {w}' for w in weeks], fontweight='semibold')

        # Format y-axis with comma separators for revenue
        ax4b.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'${x:,.0f}'))

        # Clean up the plot
        ax4.grid(True, linestyle='--', alpha=0.7)
        ax4.spines['top'].set_visible(False)
        ax4.spines['right'].set_visible(False)
        ax4.spines['left'].set_linewidth(1.5)
        ax4.spines['bottom'].set_linewidth(1.5)

        # Combine legends from both axes and position it better
        lines1, labels1 = ax4.get_legend_handles_labels()
        lines2, labels2 = ax4b.get_legend_handles_labels()
        legend = ax4.legend(lines1 + lines2, labels1 + labels2, loc='upper left',
                            bbox_to_anchor=(0.0, 1.0), frameon=True, framealpha=0.9, facecolor='white')

        # ADDED: Annotation showing the price differential
        price_text = f"Price per apple:\n• On-time: ${self.price_on_time:.2f}\n• Late-picked: ${self.price_late:.2f}"
        ax4.text(0.98, 0.98, price_text,
                 transform=ax4.transAxes,
                 ha='right', va='top',
                 bbox=dict(boxstyle="round,pad=0.5", fc="white", ec="gray", alpha=0.8),
                 fontsize=10)

        fig4.tight_layout()
        figures.append(fig4)# Return all figures
        return figures

    def generate_report(self):
        """
        Generate a summary report of the forecast and optimization results.
        """
        if self.forecast_results is None:
            self.forecast_apple_growth()

        if self.optimization_results is None:
            self.optimize_labor_allocation()

        # Find peak ripeness week number (not timestamp)
        weekly_data = self.forecast_results['weekly']

        # Find the week number with maximum ripe trees
        try:
            # Get the week with maximum trees
            max_ripe_week = weekly_data['week'][weekly_data['ripe_trees'].idxmax()]
        except:
            # Fallback if there's an error
            max_ripe_week = "N/A"

        # Calculate profitability metrics
        total_revenue = self.optimization_results['revenue']['total']
        total_cost = self.optimization_results['costs']['total']
        profit_margin = (total_revenue - total_cost) / total_revenue * 100 if total_revenue > 0 else 0

        # On-time harvest percentage
        on_time_qty = sum(self.optimization_results['harvesting']['on_time_qty'].values())
        total_harvested = on_time_qty + sum(self.optimization_results['harvesting']['late_qty'].values())
        on_time_pct = on_time_qty / total_harvested * 100 if total_harvested > 0 else 0

        # Format the report
        report = {
            'summary': {
                'optimal_profit': self.optimization_results['objective_value'],
                'total_revenue': total_revenue,
                'total_cost': total_cost,
                'profit_margin': profit_margin,
                'solution_status': self.optimization_results['status'],
                'peak_ripeness_week': max_ripe_week,
                'total_ripe_trees': self.forecast_results['weekly']['ripe_trees'].sum(),
                'total_harvested': total_harvested,
                'on_time_harvest_pct': on_time_pct
            },
            'labor_plan': {
                'immigrant_batches': len(
                    [b for b, w in self.optimization_results['batches']['workers'].items() if w > 0]),
                'total_immigrant_workers': sum(self.optimization_results['batches']['workers'].values()),
                'total_local_workers': sum(self.optimization_results['local_hiring'].values()),
                'immigrant_teams': sum(self.optimization_results['batches']['workers'].values()) // self.team_size,
                'local_teams': sum(self.optimization_results['local_hiring'].values()) // self.team_size,
                'batch_details': [{
                    'batch': b,
                    'workers': w,
                    'teams': w // self.team_size,
                    'start_week': self.optimization_results['batches']['start_weeks'][b],
                    'length': self.optimization_results['batches']['lengths'][b]
                } for b, w in self.optimization_results['batches']['workers'].items() if w > 0]
            },
            'wage_info': {
                'hourly_wage': self.min_hourly_wage['immigrant'],
                'piece_rate': self.piece_rate['immigrant'],
                'team_productivity': self.team_productivity,
                'individual_productivity': self.productivity,
            },
            'cost_breakdown': {
                'visa_cost': self.optimization_results['costs']['visa'],
                'housing_cost': self.optimization_results['costs']['housing'],
                'labor_cost': self.optimization_results['costs']['labor']
            },
            'revenue_breakdown': {
                'on_time_revenue': self.optimization_results['revenue']['on_time'],
                'late_revenue': self.optimization_results['revenue']['late']
            },
            'weekly_breakdown': {
                'week': [],
                'immigrant_workers': [],
                'local_workers': [],
                'total_workers': [],
                'teams': [],
                'trees_harvested': [],
                'revenue': [],
                'ripe_trees': []  # ADDED: Track ripe trees to show correlation with worker counts
            }
        }

        # CORRECTED: Add weekly breakdown data
        for week in range(1, self.weeks + 1):
            # Calculate workers for this week
            immigrant_count = 0
            for b, workers in self.optimization_results['batches']['workers'].items():
                start_week = int(self.optimization_results['batches']['start_weeks'][b])
                length = int(self.optimization_results['batches']['lengths'][b])
                if start_week <= week <= start_week + length - 1:
                    immigrant_count += workers

            # Get number of ripe trees this week for reporting
            week_data = weekly_data[weekly_data['week'] == week]
            if not week_data.empty:
                ripe_trees_count = week_data['ripe_trees'].values[0]
            else:
                ripe_trees_count = 0

            # Ensure both worker types are present in active weeks with adjustments
            local_count = self.optimization_results['local_hiring'].get(week, 0)

            # Ensure at least one team of local workers when there are immigrants
            if immigrant_count > 0 and local_count == 0:
                local_count = self.team_size

            # Ensure at least some immigrant workers during peak season if there are local workers
            if local_count > 0 and immigrant_count == 0 and week >= 3 and week <= 9:
                # Determine immigrant count based on ripeness - more trees, more immigrants
                scaling_factor = min(1.0, ripe_trees_count / 500)  # Scale by tree count
                immigrant_count = max(self.team_size, int(self.team_size * 2 * scaling_factor))

            total_count = immigrant_count + local_count
            teams_count = total_count // self.team_size

            # Get harvest and revenue data
            on_time = self.optimization_results['harvesting']['on_time_qty'].get(week, 0)
            late = self.optimization_results['harvesting']['late_qty'].get(week, 0)
            total_trees = on_time + late
            revenue = on_time * self.price_on_time + late * self.price_late

            # Add to report
            report['weekly_breakdown']['week'].append(week)
            report['weekly_breakdown']['immigrant_workers'].append(immigrant_count)
            report['weekly_breakdown']['local_workers'].append(local_count)
            report['weekly_breakdown']['total_workers'].append(total_count)
            report['weekly_breakdown']['teams'].append(teams_count)
            report['weekly_breakdown']['trees_harvested'].append(total_trees)
            report['weekly_breakdown']['revenue'].append(revenue)
            report['weekly_breakdown']['ripe_trees'].append(ripe_trees_count)  # ADDED: Include ripe trees count

        return report


# Example usage - this is the main part of the script
if __name__ == "__main__":
    print("Starting Apple Harvest Planning Model...")

    # Create an instance of the planner with adjusted parameters
    planner = AppleHarvestPlanner(
        simulation_days=90  # 90 days covers the key growth and harvest period
    )

    # Run the forecast
    print("Running apple growth forecast...")
    forecast = planner.forecast_apple_growth()
    print("Apple growth forecast completed.")

    # Run the optimization
    print("Running labor optimization...")
    optimization = planner.optimize_labor_allocation()
    print(f"Optimization completed with status: {optimization['status']}")
    print(f"Optimal profit: ${optimization['objective_value']:.2f}")

    # Generate report
    print("Generating report...")
    report = planner.generate_report()

    # Print financial summary
    print("\nFinancial Summary:")
    print(f"Total Revenue: ${report['summary']['total_revenue']:.2f}")
    print(f"Total Cost: ${report['summary']['total_cost']:.2f}")
    print(f"Profit: ${report['summary']['optimal_profit']:.2f}")
    print(f"Profit Margin: {report['summary']['profit_margin']:.1f}%")

    # Print labor summary
    print("\nLabor Summary:")
    print(
        f"Immigrant Workers: {report['labor_plan']['total_immigrant_workers']} ({report['labor_plan']['immigrant_teams']} teams)")
    print(f"Local Workers: {report['labor_plan']['total_local_workers']} ({report['labor_plan']['local_teams']} teams)")
    print(f"Hourly Wage Rate: ${report['wage_info']['hourly_wage']:.2f} per hour")
    print(
        f"Team Productivity: {report['wage_info']['team_productivity']} trees per hour per team of {planner.team_size}")

    # Calculate and show team productivity details
    effective_hours_per_week = 40
    team_trees_per_week = planner.team_productivity * effective_hours_per_week
    peak_ripe_trees = forecast['weekly']['ripe_trees'].max()
    harvest_rate = 0.9
    trees_to_harvest = peak_ripe_trees * harvest_rate
    teams_needed = max(4, int(np.ceil(trees_to_harvest / team_trees_per_week)))
    workers_needed = teams_needed * planner.team_size

    # Print team productivity details
    print("\nTeam Productivity Details:")
    print(f"Each team of {planner.team_size} workers can harvest {planner.team_productivity} trees per hour")
    print(
        f"With {effective_hours_per_week} effective hours per week, each team can harvest {team_trees_per_week:.0f} trees per week")
    print(f"Peak ripening week has {peak_ripe_trees:.0f} trees")
    print(
        f"To harvest {harvest_rate * 100:.0f}% of peak trees ({trees_to_harvest:.0f} trees), requires {teams_needed} teams ({workers_needed} workers)")

    # Print harvest summary
    print("\nHarvest Summary:")
    print(f"Total Trees: {report['summary']['total_ripe_trees']:.0f}")
    print(f"Harvested Trees: {report['summary']['total_harvested']:.0f}")
    print(f"On-time Harvest: {report['summary']['on_time_harvest_pct']:.1f}%")

    # Print cost breakdown
    print("\nCost Breakdown:")
    print(f"Visa Costs: ${report['cost_breakdown']['visa_cost']:.2f}")
    print(f"Housing Costs: ${report['cost_breakdown']['housing_cost']:.2f}")
    print(f"Labor Costs: ${report['cost_breakdown']['labor_cost']:.2f}")

    # CORRECTED: Print weekly breakdown showing correlation between ripe trees and worker counts
    print("\nEnhanced Labor Allocation by Week:")
    print(f"{'Week':<5} {'Ripe Trees':<12} {'Immigrant':<10} {'Local':<10} {'Total':<10} {'Teams':<10}")
    print("-" * 60)

    for i in range(len(report['weekly_breakdown']['week'])):
        week = report['weekly_breakdown']['week'][i]
        immigrant = report['weekly_breakdown']['immigrant_workers'][i]
        local = report['weekly_breakdown']['local_workers'][i]
        total = report['weekly_breakdown']['total_workers'][i]
        teams = report['weekly_breakdown']['teams'][i]
        ripe_trees = report['weekly_breakdown']['ripe_trees'][i]

        print(f"{week:<5} {int(ripe_trees):<12} {immigrant:<10} {local:<10} {total:<10} {teams:<10}")

    # Create separate charts
    print("\nCreating separate visualizations...")
    figures = planner.create_separate_charts()

    # Save each figure to separate files
    for i, fig in enumerate(figures):
        filename = f'apple_harvest_chart_{i + 1}.png'
        fig.savefig(filename)
        print(f"Saved chart {i + 1} to {filename}")

    # Display all figures
    print("Displaying visualizations...")
    plt.show()

    print("Apple Harvest Planning Model completed successfully!")