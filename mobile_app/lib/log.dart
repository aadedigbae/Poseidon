import 'package:flutter/material.dart';
import 'package:mobile_app/firestore/sensor_reading_service.dart';
import 'package:intl/intl.dart';

class ActionLogWidget extends StatefulWidget {
  const ActionLogWidget({Key? key}) : super(key: key);

  @override
  State<ActionLogWidget> createState() => _ActionLogWidgetState();
}

class _ActionLogWidgetState extends State<ActionLogWidget> {
  final SensorReadingService _service = SensorReadingService();
  Stream<Map<String, dynamic>?>? _actionLogStream;

  @override
  void initState() {
    super.initState();
    _actionLogStream = _service.streamLatestActionLog();
  }

  @override
  Widget build(BuildContext context) {
    return StreamBuilder<Map<String, dynamic>?>(
      stream: _actionLogStream,
      builder: (context, snapshot) {
        final actionData = snapshot.data ?? {};
        print('Action Log Data: $actionData');
        if (!snapshot.hasData || snapshot.data == null) {
          return const SizedBox.shrink();
        }

        return GestureDetector(
          onTap: () => _showActionLogBottomSheet(actionData),
          child: _buildActionLogButton(actionData),
        );
      },
    );
  }

  Widget _buildActionLogButton(Map<String, dynamic> actionData) {
    final alertLevel = actionData['alert_level'] ?? 'INFO';
    final config = _getAlertConfig(alertLevel);

    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 12),
      decoration: BoxDecoration(
        color: config['backgroundColor'],
        borderRadius: BorderRadius.circular(12),
        border: Border.all(color: config['borderColor'], width: 1),
      ),
      child: Row(
        children: [
          Icon(config['icon'], color: config['iconColor'], size: 20),
          const SizedBox(width: 8),
          Text(
            'View Recommended Actions',
            style: TextStyle(
              fontSize: 14,
              fontWeight: FontWeight.w600,
              color: config['textColor'],
            ),
          ),
          const Spacer(),
          Icon(Icons.arrow_forward_ios, size: 14, color: config['iconColor']),
        ],
      ),
    );
  }

  void _showActionLogBottomSheet(Map<String, dynamic> actionData) {
    final alertLevel = actionData['alert_level'] ?? 'INFO';
    final immediateActions = List<String>.from(
      actionData['immediate_actions'] ?? [],
    );
    final investigationActions = List<String>.from(
      actionData['investigation_actions'] ?? [],
    );
    final preventiveActions = List<String>.from(
      actionData['preventive_actions'] ?? [],
    );

    final config = _getAlertConfig(alertLevel);

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
          child: SingleChildScrollView(
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
                        decoration: BoxDecoration(
                          color: config['iconColor'],
                          shape: BoxShape.circle,
                        ),
                        child: Icon(
                          config['icon'],
                          color: Colors.white,
                          size: 20,
                        ),
                      ),
                      const SizedBox(width: 12),
                      Text(
                        '${alertLevel.toUpperCase()} ALERT',
                        style: const TextStyle(
                          fontSize: 20,
                          fontWeight: FontWeight.w600,
                          color: Colors.black,
                        ),
                      ),
                      const Spacer(),
                      GestureDetector(
                        onTap: () => Navigator.pop(context),
                        child: const Icon(
                          Icons.close,
                          color: Colors.black54,
                          size: 24,
                        ),
                      ),
                    ],
                  ),

                  const SizedBox(height: 24),

                  // Immediate Actions
                  if (immediateActions.isNotEmpty) ...[
                    _buildSectionHeader(
                      'IMMEDIATE ACTIONS:',
                      Icons.flash_on,
                      const Color(0xFFF54101),
                    ),
                    const SizedBox(height: 8),
                    ...immediateActions.map(
                      (action) => _buildBulletPoint(action),
                    ),
                    const SizedBox(height: 16),
                  ],

                  // Investigation Actions
                  if (investigationActions.isNotEmpty) ...[
                    _buildSectionHeader(
                      'INVESTIGATION REQUIRED:',
                      Icons.search,
                      const Color(0xFFFFA726),
                    ),
                    const SizedBox(height: 8),
                    ...investigationActions.map(
                      (action) => _buildBulletPoint(action),
                    ),
                    const SizedBox(height: 16),
                  ],

                  // Preventive Actions
                  if (preventiveActions.isNotEmpty) ...[
                    _buildSectionHeader(
                      'PREVENTIVE MEASURES:',
                      Icons.shield_outlined,
                      const Color(0xFF66BB6A),
                    ),
                    const SizedBox(height: 8),
                    ...preventiveActions.map(
                      (action) => _buildBulletPoint(action),
                    ),
                  ],

                  const SizedBox(height: 16),

                  // Timestamp
                  if (actionData['timestamp'] != null) ...[
                    Divider(color: Colors.grey.shade300),
                    const SizedBox(height: 8),
                    Row(
                      children: [
                        Icon(
                          Icons.access_time,
                          size: 16,
                          color: Colors.grey.shade600,
                        ),
                        const SizedBox(width: 8),
                        Text(
                          'Generated: ${_formatTimestamp(actionData['timestamp'])}',
                          style: TextStyle(
                            fontSize: 12,
                            color: Colors.grey.shade600,
                          ),
                        ),
                      ],
                    ),
                  ],
                ],
              ),
            ),
          ),
        ),
      ),
    );
  }

  Widget _buildSectionHeader(String title, IconData icon, Color color) {
    return Row(
      children: [
        Icon(icon, size: 18, color: color),
        const SizedBox(width: 8),
        Text(
          title,
          style: TextStyle(
            fontSize: 16,
            fontWeight: FontWeight.w600,
            color: color,
          ),
        ),
      ],
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

  Map<String, dynamic> _getAlertConfig(String alertLevel) {
    switch (alertLevel.toUpperCase()) {
      case 'CRITICAL':
      case 'HIGH':
        return {
          'icon': Icons.priority_high,
          'iconColor': const Color(0xFFF54101),
          'backgroundColor': const Color(0xFFFFEBEE),
          'borderColor': const Color(0xFFF54101),
          'textColor': const Color(0xFFF54101),
        };
      case 'MEDIUM':
      case 'WARNING':
        return {
          'icon': Icons.warning_amber_rounded,
          'iconColor': const Color(0xFFFFA726),
          'backgroundColor': const Color(0xFFFFF3E0),
          'borderColor': const Color(0xFFFFA726),
          'textColor': const Color(0xFFF57C00),
        };
      case 'LOW':
      case 'INFO':
      default:
        return {
          'icon': Icons.info_outline,
          'iconColor': const Color(0xFF2196F3),
          'backgroundColor': const Color(0xFFE3F2FD),
          'borderColor': const Color(0xFF2196F3),
          'textColor': const Color(0xFF1976D2),
        };
    }
  }

  String _formatTimestamp(dynamic timestamp) {
    try {
      DateTime dateTime;
      if (timestamp is DateTime) {
        dateTime = timestamp;
      } else {
        // Firestore Timestamp
        dateTime = timestamp.toDate();
      }
      return DateFormat('MMM dd, yyyy • hh:mm a').format(dateTime);
    } catch (e) {
      return 'Unknown';
    }
  }
}
