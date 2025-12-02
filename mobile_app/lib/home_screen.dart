import 'package:flutter/material.dart';
import 'dart:math' as math;

import 'package:mobile_app/firestore/sensor_reading_service.dart';
import 'package:mobile_app/forecast.dart';
import 'package:mobile_app/log.dart';

class FishFarmHomeScreen extends StatefulWidget {
  const FishFarmHomeScreen({super.key});

  @override
  State<FishFarmHomeScreen> createState() => _FishFarmHomeScreenState();
}

class _FishFarmHomeScreenState extends State<FishFarmHomeScreen> {
  String selectedForecastPeriod = '4 hrs';
  // Map<String, dynamic>? _latestReading;
  final SensorReadingService _service = SensorReadingService();
  Stream<Map<String, dynamic>?>? _sensorStream;

  @override
  void initState() {
    super.initState();
    // Initialize the stream in initState
    _sensorStream = _service.streamLatestSensorReading();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: StreamBuilder<Map<String, dynamic>?>(
        stream: _sensorStream,
        builder: (context, snapshot) {
          return SingleChildScrollView(
            child: Column(
              children: [
                _buildHeader(),
                Padding(
                  padding: const EdgeInsets.all(16.0),
                  child: Column(
                    children: [
                      // Fish Mortality Risk Card
                      _buildMortalityRiskCard(snapshot),
                      const SizedBox(height: 20),
                      // Mortality Forecast Section
                      // _buildMortalityForecast(),
                      MortalityForecastWidget(),
                      const SizedBox(height: 20),
                      // Action Card
                      // _buildActionCard(),
                      ActionLogWidget(),
                      const SizedBox(height: 100),
                    ],
                  ),
                ),
              ],
            ),
          );
        },
      ),
    );
  }

  Widget _buildHeader() {
    return Container(
      height: 320,
      width: double.infinity,
      decoration: BoxDecoration(
        image: DecorationImage(
          image: AssetImage('assets/fish_farm_background.png'),
          fit: BoxFit.cover,
        ),
      ),
      child: Container(
        decoration: BoxDecoration(
          gradient: LinearGradient(
            begin: Alignment.topCenter,
            end: Alignment.bottomCenter,
            colors: [
              Colors.black.withOpacity(0.3),
              Colors.black.withOpacity(0.2),
              const Color(0xFFF8F9FA).withOpacity(0.7),
              const Color(0xFFF8F9FA),
            ],
            stops: const [0.0, 0.4, 0.85, 1.0],
          ),
        ),
        child: SafeArea(
          child: Padding(
            padding: const EdgeInsets.all(20.0),
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                // Farm ID
                Row(
                  children: [
                    Container(
                      width: 40,
                      height: 40,
                      decoration: BoxDecoration(
                        shape: BoxShape.circle,
                        color: Colors.white,
                        image: DecorationImage(
                          image: AssetImage('assets/farm_icon.png'),
                          fit: BoxFit.cover,
                        ),
                      ),
                    ),
                    const SizedBox(width: 12),
                    const Text(
                      'pond1',
                      style: TextStyle(
                        color: Colors.white,
                        fontSize: 18,
                        fontWeight: FontWeight.w600,
                        fontFamily: 'Geist',
                        letterSpacing: -2,
                      ),
                    ),
                  ],
                ),

                const SizedBox(height: 16),

                // Farm Name
                const Text(
                  'Folamot Farm',
                  style: TextStyle(
                    color: Colors.white,
                    fontSize: 40,
                    fontWeight: FontWeight.bold,
                    letterSpacing: -2,
                    fontFamily: 'Geist',
                  ),
                ),

                const SizedBox(height: 6),

                Row(
                  children: [
                    const Text(
                      '2025-10-18 . 10:00:00',
                      style: TextStyle(
                        color: Colors.white,
                        fontSize: 18,
                        fontWeight: FontWeight.w500,
                        letterSpacing: -2,
                      ),
                    ),
                    const SizedBox(width: 12),
                    Container(
                      padding: const EdgeInsets.symmetric(
                        horizontal: 14,
                        vertical: 6,
                      ),
                      decoration: BoxDecoration(
                        color: const Color(0xFFCFEEFF),
                        borderRadius: BorderRadius.circular(20),
                      ),
                      child: Row(
                        children: const [
                          Text(
                            'See All History',
                            style: TextStyle(
                              color: Color(0XFF249EDF),
                              fontSize: 13,
                              fontWeight: FontWeight.w600,
                              letterSpacing: -1,
                            ),
                          ),
                          SizedBox(width: 6),
                          Icon(
                            Icons.arrow_forward,
                            color: Color(0XFF249EDF),
                            size: 14,
                          ),
                        ],
                      ),
                    ),
                  ],
                ),
              ],
            ),
          ),
        ),
      ),
    );
  }

  Widget _buildMortalityRiskCard(
    AsyncSnapshot<Map<String, dynamic>?> snapshot,
  ) {
    final reading = snapshot.data;
    final riskLabel = snapshot.hasData && reading != null
        ? '${reading['risk_label'] ?? "N/A"}'
        : 'Loading...';
    final riskProbability = snapshot.hasData && reading != null
        ? (getHighestRisk(reading)['probability'] as double? ?? 0.0)
        : 0.0;
    print('Latest Reading: $reading');

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
            'Fish Mortality Risk',
            style: TextStyle(
              fontSize: 20,
              fontWeight: FontWeight.w500,
              color: Colors.black,
            ),
          ),

          const SizedBox(height: 16),

          Row(
            children: [
              Expanded(
                child: _buildMetric(
                  'Dis. Oxygen:',
                  snapshot.hasData && reading != null
                      ? '${reading['predicted_do'] ?? "N/A"}'
                      : 'Loading...',
                  Icons.water_drop,
                ),
              ),
              const SizedBox(width: 16),
              Expanded(
                child: _buildMetric(
                  'pH Level:',
                  snapshot.hasData && reading != null
                      ? '${reading['pH'] ?? "N/A"}'
                      : 'Loading...',
                  Icons.circle,
                ),
              ),
            ],
          ),

          const SizedBox(height: 16),

          // Metrics Row 2
          Row(
            children: [
              Expanded(
                child: _buildMetric(
                  'Temperature:',
                  snapshot.hasData && reading != null
                      ? '${reading['temperature'] ?? "N/A"}'
                      : 'Loading...',
                  Icons.thermostat,
                ),
              ),
              const SizedBox(width: 16),
              Expanded(
                child: _buildMetric(
                  'Turbidity:',
                  snapshot.hasData && reading != null
                      ? '${reading['turbidity_proxy'] ?? "N/A"} mg/L'
                      : 'Loading...',
                  Icons.water_drop,
                ),
              ),
            ],
          ),

          const SizedBox(height: 16),

          Row(
            children: [
              Expanded(
                child: _buildMetric(
                  'Ammonia:',
                  snapshot.hasData && reading != null
                      ? '${reading['predicted_nh3'] ?? "N/A"}'
                      : 'Loading...',
                  Icons.science,
                ),
              ),
            ],
          ),

          const SizedBox(height: 30),

          // Linear Gradient Bar
          _buildLinearGradient(
            riskProbability is double ? riskProbability : 0.0,
          ),

          const SizedBox(height: 20),

          // Risk Label (NO TAP)
          Center(
            child: Column(
              children: [
                const Text(
                  'Mortality Risk:',
                  style: TextStyle(
                    fontSize: 14,
                    fontWeight: FontWeight.w400,
                    color: Colors.black,
                  ),
                ),
                const SizedBox(height: 4),
                Row(
                  mainAxisAlignment: MainAxisAlignment.center,
                  crossAxisAlignment: CrossAxisAlignment.baseline,
                  textBaseline: TextBaseline.alphabetic,
                  children: [
                    Text(
                      riskLabel,
                      style: TextStyle(
                        fontSize: 48,
                        fontWeight: FontWeight.w800,
                        color: Colors.black,
                        letterSpacing: 1,
                      ),
                    ),
                    Text(
                      ' ${riskProbability.toStringAsFixed(2)}',
                      style: TextStyle(
                        fontSize: 32,
                        fontWeight: FontWeight.w600,
                        color: Color(0xFFF63501),
                      ),
                    ),
                  ],
                ),
              ],
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildMetric(String label, String value, IconData icon) {
    return Row(
      children: [
        Container(
          padding: const EdgeInsets.all(6),
          decoration: BoxDecoration(
            color: const Color(0xFF6EC1E4).withOpacity(0.15),
            shape: BoxShape.circle,
          ),
          child: Icon(icon, size: 14, color: const Color(0xFF6EC1E4)),
        ),
        const SizedBox(width: 10),
        Expanded(
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Text(
                label,
                style: const TextStyle(
                  fontSize: 13,
                  fontWeight: FontWeight.w400,
                  color: Colors.black54,
                ),
              ),
              const SizedBox(height: 2),
              Text(
                value,
                style: const TextStyle(
                  fontSize: 16,
                  fontWeight: FontWeight.w700,
                  color: Colors.black87,
                ),
              ),
            ],
          ),
        ),
      ],
    );
  }

  Widget _buildLinearGradient(double riskProbability) {
    return Column(
      children: [
        Stack(
          clipBehavior: Clip.none,
          alignment: Alignment.center,
          children: [
            Container(
              height: 45,
              decoration: BoxDecoration(
                borderRadius: BorderRadius.circular(8),
                gradient: const LinearGradient(
                  colors: [
                    Color(0xFF7ED321),
                    Color(0xFFD4E157),
                    Color(0xFFFFCA28),
                    Color(0xFFFF9800),
                    Color(0xFFFF5722),
                    Color(0xFFE53935),
                  ],
                  stops: [0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
                ),
              ),
            ),
            // Triangle indicator at 83% position
            Positioned(
              left:
                  MediaQuery.of(context).size.width * riskProbability -
                  40, // Adjust based on padding
              bottom: -10,
              child: CustomPaint(
                painter: TrianglePainter(),
                child: const SizedBox(width: 16, height: 12),
              ),
            ),
          ],
        ),
      ],
    );
  }

  Widget _buildActionCard() {
    return Container(
      padding: const EdgeInsets.all(16),
      decoration: BoxDecoration(
        color: const Color(0xFF1E1E1E),
        borderRadius: BorderRadius.circular(50),
      ),
      child: Row(
        children: [
          Container(
            padding: const EdgeInsets.all(14),
            decoration: const BoxDecoration(
              color: Color(0xFFF54101),
              shape: BoxShape.circle,
            ),
            child: const Icon(
              Icons.priority_high,
              color: Colors.white,
              size: 24,
            ),
          ),
          const SizedBox(width: 16),
          Expanded(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: const [
                Text(
                  'Boost Aeration',
                  style: TextStyle(
                    fontSize: 18,
                    fontWeight: FontWeight.w600,
                    color: Colors.white,
                  ),
                ),
                SizedBox(height: 4),
                Text(
                  'Dissolved oxygen dropping rapidly (6.2 → 4.1 mg/L i...',
                  style: TextStyle(
                    fontSize: 14,
                    fontWeight: FontWeight.w600,
                    color: Colors.white70,
                  ),
                  maxLines: 1,
                  overflow: TextOverflow.ellipsis,
                ),
              ],
            ),
          ),
          const SizedBox(width: 1),
          GestureDetector(
            onTap: () {
              _showRiskBottomSheet();
            },
            child: Container(
              padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 10),
              decoration: BoxDecoration(
                color: const Color(0xFF424242),
                borderRadius: BorderRadius.circular(50),
              ),
              child: Row(
                children: const [
                  Text(
                    'View All Insight',
                    style: TextStyle(
                      fontSize: 10,
                      fontWeight: FontWeight.w600,
                      color: Colors.white,
                    ),
                  ),
                  SizedBox(width: 6),
                  Icon(Icons.arrow_forward, color: Colors.white, size: 14),
                ],
              ),
            ),
          ),
        ],
      ),
    );
  }

  void _showRiskBottomSheet() {
    showModalBottomSheet(
      context: context,
      isScrollControlled: true,
      backgroundColor: Colors.transparent,
      builder: (context) => Container(
        decoration: const BoxDecoration(
          color: Colors.white,
          borderRadius: BorderRadius.only(
            topLeft: Radius.circular(40),
            topRight: Radius.circular(40),
          ),
        ),
        child: SafeArea(
          child: Padding(
            padding: const EdgeInsets.all(24),
            child: Column(
              mainAxisSize: MainAxisSize.min,
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                // Header
                Row(
                  children: [
                    Container(
                      padding: const EdgeInsets.all(8),
                      decoration: const BoxDecoration(
                        color: Color(0xFFF54101),
                        shape: BoxShape.circle,
                      ),
                      child: const Icon(
                        Icons.priority_high,
                        color: Colors.white,
                        size: 20,
                      ),
                    ),
                    const SizedBox(width: 12),
                    const Text(
                      'HIGH RISK',
                      style: TextStyle(
                        fontSize: 20,
                        fontWeight: FontWeight.w600,
                        color: Colors.black,
                      ),
                    ),
                    const Spacer(),
                    GestureDetector(
                      onTap: () {
                        Navigator.pop(context);
                      },
                      child: const Icon(
                        Icons.close,
                        color: Colors.black54,
                        size: 24,
                      ),
                    ),
                  ],
                ),

                const SizedBox(height: 8),

                // Why This Is Happening
                const Text(
                  'WHY THIS IS HAPPENING:',
                  style: TextStyle(
                    fontSize: 16,
                    fontWeight: FontWeight.w400,
                    color: Colors.black,
                  ),
                ),

                _buildBulletPoint(
                  'PRIMARY FACTOR - Dissolved oxygen dropping rapidly (6.2 → 4.1 mg/L in 2 hours)',
                ),
                _buildBulletPoint(
                  'SECONDARY FACTOR - Ammonia building up (0.25 mg/L)',
                ),
                _buildBulletPoint(
                  'CONTRIBUTING FACTOR - Water temperature elevated (29°C)',
                ),

                const SizedBox(height: 8),

                // Recommended Actions
                const Text(
                  'RECOMMENDED ACTIONS:',
                  style: TextStyle(
                    fontSize: 16,
                    fontWeight: FontWeight.w400,
                    color: Colors.black,
                  ),
                ),

                const SizedBox(height: 8),

                _buildBulletPoint('Boost aeration to maximum capacity'),
                _buildBulletPoint('Reduce feeding by 50% for next cycle'),
                _buildBulletPoint('Increase water circulation'),
              ],
            ),
          ),
        ),
      ),
    );
  }

  Widget _buildBulletPoint(String text) {
    return Padding(
      padding: const EdgeInsets.only(bottom: 8.0),
      child: Row(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          const Text(
            '• ',
            style: TextStyle(
              fontSize: 14,
              fontWeight: FontWeight.w600,
              color: Colors.black87,
            ),
          ),
          Expanded(
            child: Text(
              text,
              style: const TextStyle(
                fontSize: 14,
                fontWeight: FontWeight.w400,
                color: Colors.black,
                height: 1.4,
              ),
            ),
          ),
        ],
      ),
    );
  }
}

// Custom Painter for Triangle Indicator
class TrianglePainter extends CustomPainter {
  @override
  void paint(Canvas canvas, Size size) {
    final paint = Paint()
      ..color = const Color(0xFFE53935)
      ..style = PaintingStyle.fill;

    final path = Path();
    path.moveTo(size.width / 2, 0);
    path.lineTo(0, size.height);
    path.lineTo(size.width, size.height);
    path.close();

    canvas.drawPath(path, paint);
  }

  @override
  bool shouldRepaint(covariant CustomPainter oldDelegate) => false;
}

Map<String, dynamic> getHighestRisk(Map<String, dynamic> reading) {
  Map<String, double> probabilities = {
    'HIGH': reading['prob_HIGH'] ?? 0.0,
    'MEDIUM': reading['prob_MEDIUM'] ?? 0.0,
    'LOW': reading['prob_LOW'] ?? 0.0,
  };

  var maxEntry = probabilities.entries.reduce(
    (a, b) => a.value > b.value ? a : b,
  );

  return {'label': maxEntry.key, 'probability': maxEntry.value};
}
