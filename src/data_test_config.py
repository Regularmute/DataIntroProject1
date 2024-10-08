configs = [{
    'keep': [
        'day_of_week_sin', 'day_of_week_cos', 'hour_sin', 'hour_cos',
        'temperature', 'export', 'month_sin', 'month_cos', 'CO2', 'production', 'electricity_cost', 'date'
    ],
    'drop': [
        'hour', 'day', 'month', 'year', 'consumption', 'hydro', 'district', 'day_of_week'
    ]
}, {
    'keep': [
        'day_of_week_sin', 'day_of_week_cos', 'hour_sin', 'hour_cos',
        'temperature', 'month_sin', 'month_cos', 'CO2', 'production', 'electricity_cost', 'date'
    ],
    'drop': [
        'hour', 'day', 'month', 'year', 'consumption', 'hydro', 'district', 'export', 'day_of_week'
    ]
}, {
    'keep': [
        'day_of_week_sin', 'day_of_week_cos', 'hour_sin', 'hour_cos',
        'month_sin', 'month_cos', 'CO2', 'production', 'electricity_cost', 'date'
    ],
    'drop': [
        'hour', 'day', 'month', 'year', 'consumption', 'hydro', 'district', 'export', 'temperature', 'day_of_week'
    ]
}, {
    'keep': [
        'day_of_week_sin', 'day_of_week_cos', 'hour_sin', 'hour_cos',
        'temperature', 'month_sin', 'month_cos', 'CO2', 'electricity_cost', 'date'
    ],
    'drop': [
        'hour', 'day', 'month', 'year', 'consumption', 'hydro', 'district', 'export', 'production', 'day_of_week'
    ]
}, {
    'keep': [
        'day_of_week_sin', 'hour_sin', 'hour_cos',
        'temperature', 'month_sin', 'month_cos', 'CO2', 'electricity_cost', 'date'
    ],
    'drop': [
        'hour', 'day', 'month', 'year', 'consumption', 'hydro', 'district', 'export', 'production', 'day_of_week_cos', 'day_of_week'
    ]
}, {
    'keep': [
        'day_of_week_sin', 'hour_sin',
        'temperature', 'month_sin', 'month_cos', 'CO2', 'electricity_cost', 'date'
    ],
    'drop': [
        'hour', 'day', 'month', 'year', 'consumption', 'hydro', 'district', 'export', 'production', 'day_of_week_cos', 'hour_cos', 'day_of_week'
    ]
}, {
    'keep': [
        'day_of_week_sin',  'hour_cos', 'production', 'wind', 'day_of_week_cos'
        'temperature', 'month_sin', 'month_cos', 'CO2', 'electricity_cost', 'date'
    ],
    'drop': [
        'hour', 'day', 'month', 'year', 'consumption',
        'hydro', 'district', 'export', 'day_of_week',
        'hour_sin',
    ]
}
]
