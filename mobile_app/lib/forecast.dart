import 'package:flutter/material.dart';
import 'package:mobile_app/firestore/sensor_reading_service.dart';

class MortalityForecastWidget extends StatefulWidget {
  @override
  State<MortalityForecastWidget> createState() =>
      _MortalityForecastWidgetState();
}

class _MortalityForecastWidgetState extends State<MortalityForecastWidget> {
  final SensorReadingService _service = SensorReadingService();
  Stream<Map<String, dynamic>?>? _sensorStream;

  // Map periods to Firestore field names
  String _selectedPeriod = '1 hr';

  final Map<String, String> _periodToFieldMap = {
    '1 hr': 'forecast_+1h',
    '6 hrs': 'forecast_+6h',
    '24 hrs': 'forecast_+24h',
    '3 days': 'forecast_+3d',
  };

  @override
  void initState() {
    super.initState();
    _sensorStream = _service.streamLatestForecast();
  }

  @override
  Widget build(BuildContext context) {
    return StreamBuilder<Map<String, dynamic>?>(
      stream: _sensorStream,
      builder: (context, snapshot) {
        final forecastData = snapshot.data;
        print('Forecast Data: $forecastData');

        if (snapshot.connectionState == ConnectionState.waiting) {
          return _buildLoadingState();
        }

        if (snapshot.hasError) {
          return _buildErrorState(snapshot.error.toString());
        }

        if (forecastData == null) {
          return _buildEmptyState();
        }

        return _buildMortalityForecast(forecastData);
      },
    );
  }

  Widget _buildMortalityForecast(Map<String, dynamic> forecastData) {
    // Extract risk trend
    final riskTrend = forecastData['risk_trend'] ?? 'UNKNOWN';
    final color = _getRiskColor(riskTrend);

    // Get forecast value for selected period
    final fieldName = _periodToFieldMap[_selectedPeriod]!;
    final forecastValue = forecastData[fieldName] ?? 0.0;

    // Convert to percentage with 2 decimal places
    final percentage = (forecastValue * 100).toStringAsFixed(2);

    return Container(
      padding: const EdgeInsets.all(24),
      decoration: BoxDecoration(
        color: Colors.white,
        borderRadius: BorderRadius.circular(20),
        boxShadow: [
          BoxShadow(
            color: Colors.black.withOpacity(0.05),
            blurRadius: 10,
            offset: const Offset(0, 2),
          ),
        ],
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          const Text(
            'Mortality Forecast',
            style: TextStyle(
              fontSize: 20,
              fontWeight: FontWeight.w600,
              color: Colors.black87,
            ),
          ),
          const SizedBox(height: 20),
          Row(
            children: [
              _buildPeriodButton('1 hr'),
              const SizedBox(width: 10),
              _buildPeriodButton('6 hrs'),
              const SizedBox(width: 10),
              _buildPeriodButton('24 hrs'),
              const SizedBox(width: 10),
              _buildPeriodButton('3 days'),
            ],
          ),
          const SizedBox(height: 24),
          Row(
            crossAxisAlignment: CrossAxisAlignment.center,
            children: [
              Container(
                width: 18,
                height: 80,
                decoration: BoxDecoration(
                  color: color,
                  borderRadius: BorderRadius.circular(4),
                ),
              ),
              const SizedBox(width: 16),
              Expanded(
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    const Text(
                      'Risk Trend:',
                      style: TextStyle(
                        fontSize: 14,
                        fontWeight: FontWeight.w400,
                        color: Colors.black,
                      ),
                    ),
                    const SizedBox(height: 4),
                    Row(
                      crossAxisAlignment: CrossAxisAlignment.baseline,
                      textBaseline: TextBaseline.alphabetic,
                      children: [
                        Text(
                          riskTrend.toUpperCase(),
                          style: const TextStyle(
                            fontSize: 32,
                            fontWeight: FontWeight.w800,
                            color: Colors.black,
                          ),
                        ),
                        Text(
                          ' . $percentage%',
                          style: TextStyle(
                            fontSize: 24,
                            fontWeight: FontWeight.w600,
                            color: color,
                          ),
                        ),
                      ],
                    ),
                  ],
                ),
              ),
            ],
          ),
        ],
      ),
    );
  }

  Widget _buildPeriodButton(String period) {
    final isSelected = _selectedPeriod == period;
    return GestureDetector(
      onTap: () {
        setState(() {
          _selectedPeriod = period;
        });
      },
      child: Container(
        padding: const EdgeInsets.symmetric(horizontal: 20, vertical: 12),
        decoration: BoxDecoration(
          color: isSelected ? const Color(0xFFB3E5FC) : const Color(0xFFF5F5F5),
          borderRadius: BorderRadius.circular(25),
        ),
        child: Text(
          period,
          style: TextStyle(
            fontSize: 12,
            fontWeight: FontWeight.w600,
            color: isSelected ? const Color(0xFF01579B) : Colors.black,
          ),
        ),
      ),
    );
  }

  Widget _buildLoadingState() {
    return Container(
      padding: const EdgeInsets.all(24),
      decoration: BoxDecoration(
        color: Colors.white,
        borderRadius: BorderRadius.circular(20),
        boxShadow: [
          BoxShadow(
            color: Colors.black.withOpacity(0.05),
            blurRadius: 10,
            offset: const Offset(0, 2),
          ),
        ],
      ),
      child: const Center(child: CircularProgressIndicator()),
    );
  }

  Widget _buildErrorState(String error) {
    return Container(
      padding: const EdgeInsets.all(24),
      decoration: BoxDecoration(
        color: Colors.white,
        borderRadius: BorderRadius.circular(20),
        boxShadow: [
          BoxShadow(
            color: Colors.black.withOpacity(0.05),
            blurRadius: 10,
            offset: const Offset(0, 2),
          ),
        ],
      ),
      child: Text('Error: $error'),
    );
  }

  Widget _buildEmptyState() {
    return Container(
      padding: const EdgeInsets.all(24),
      decoration: BoxDecoration(
        color: Colors.white,
        borderRadius: BorderRadius.circular(20),
        boxShadow: [
          BoxShadow(
            color: Colors.black.withOpacity(0.05),
            blurRadius: 10,
            offset: const Offset(0, 2),
          ),
        ],
      ),
      child: const Text('No forecast data available'),
    );
  }

  Color _getRiskColor(String riskTrend) {
    switch (riskTrend.toLowerCase()) {
      case 'increasing':
        return const Color(0xFFF72D00); // Red for increasing risk
      case 'stable':
        return const Color(0xFFFFA726); // Orange for stable
      case 'decreasing':
        return const Color(0xFF66BB6A); // Green for decreasing risk
      default:
        return const Color(0xFFF72D00);
    }
  }
}
