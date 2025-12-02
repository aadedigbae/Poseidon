import 'package:cloud_firestore/cloud_firestore.dart';

class SensorReadingService {
  final FirebaseFirestore _firestore = FirebaseFirestore.instance;

  /// Fetches the most recent sensor reading from pond_1
  // Future<Map<String, dynamic>?> getLatestSensorReading() async {
  //   try {
  //     final querySnapshot = await _firestore
  //         .collection('ponds')
  //         .doc('pond1')
  //         .collection('risk_predictions')
  //         .orderBy(
  //           'timestamp',
  //           descending: true,
  //         ) // Assumes you have a timestamp field
  //         .limit(1)
  //         .get();

  //     if (querySnapshot.docs.isNotEmpty) {
  //       final doc = querySnapshot.docs.first;
  //       return {'id': doc.id, ...doc.data()};
  //     }
  //     return null;
  //   } catch (e) {
  //     print('Error fetching latest sensor reading: $e');
  //     rethrow;
  //   }
  // }

  /// Alternative: Stream the latest sensor reading (real-time updates)
  Stream<Map<String, dynamic>?> streamLatestSensorReading() {
    return _firestore
        .collection('ponds')
        .doc('pond1')
        .collection('risk_predictions')
        .orderBy('timestamp', descending: true)
        .limit(1)
        .snapshots()
        .map((snapshot) {
          if (snapshot.docs.isNotEmpty) {
            final doc = snapshot.docs.first;
            return {'id': doc.id, ...doc.data()};
          }
          return null;
        });
  }

  Stream<Map<String, dynamic>?> streamLatestForecast() {
    return _firestore
        .collection('ponds')
        .doc('pond1')
        .collection('forecasts')
        .orderBy('timestamp', descending: true)
        .limit(1)
        .snapshots()
        .map((snapshot) {
          if (snapshot.docs.isNotEmpty) {
            final doc = snapshot.docs.first;
            return {'id': doc.id, ...doc.data()};
          }
          return null;
        });
  }

  Stream<Map<String, dynamic>?> streamLatestActionLog() {
    return _firestore
        .collection('ponds')
        .doc('pond1')
        .collection('actions_log')
        .orderBy('timestamp', descending: true)
        .limit(1)
        .snapshots()
        .map((snapshot) {
          if (snapshot.docs.isNotEmpty) {
            final doc = snapshot.docs.first;
            return {'id': doc.id, ...doc.data()};
          }
          return null;
        });
  }
}
