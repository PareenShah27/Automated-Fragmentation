"""
Data Loader Module
Loads and manages database schemas with schema-specific query patterns.
Now includes query patterns for E-Commerce, Healthcare, and University schemas.
"""

from typing import Dict, List, Any
import random


class DataLoader:
    """Manages schema loading, workload generation, and schema-specific query patterns."""
    
    SCHEMAS = {
        "E-Commerce": {
            "tables": ["Customers", "Orders", "Products", "Inventory", "Payments"],
            "attributes": {
                "Customers": ["customer_id", "name", "email", "region", "created_at", "credit_limit"],
                "Orders": ["order_id", "customer_id", "product_id", "amount", "order_date", "status"],
                "Products": ["product_id", "name", "category", "price", "stock", "supplier_id"],
                "Inventory": ["product_id", "warehouse_id", "quantity", "location", "last_updated"],
                "Payments": ["payment_id", "order_id", "amount", "method", "transaction_date"]
            }
        },
        "Healthcare": {
            "tables": ["Patients", "Doctors", "Appointments", "Medical_Records", "Prescriptions"],
            "attributes": {
                "Patients": ["patient_id", "name", "dob", "region", "insurance", "medical_history"],
                "Doctors": ["doctor_id", "name", "specialization", "hospital", "experience"],
                "Appointments": ["appt_id", "patient_id", "doctor_id", "date", "status", "notes"],
                "Medical_Records": ["record_id", "patient_id", "diagnosis", "treatment", "date"],
                "Prescriptions": ["prescription_id", "patient_id", "doctor_id", "medication", "dosage"]
            }
        },
        "University": {
            "tables": ["Students", "Faculty", "Courses", "Enrollments", "Grades"],
            "attributes": {
                "Students": ["student_id", "name", "dob", "department", "enrollment_date"],
                "Faculty": ["faculty_id", "name", "specialization", "department", "hire_date"],
                "Courses": ["course_id", "name", "faculty_id", "credits", "department"],
                "Enrollments": ["enrollment_id", "student_id", "course_id", "semester", "status"],
                "Grades": ["grade_id", "enrollment_id", "midterm", "final", "gpa"]
            }
        }
    }
    
    # SCHEMA-SPECIFIC QUERY PATTERNS
    QUERY_PATTERNS = {
        "E-Commerce": [
            {
                'query_type': 'SELECT',
                'description': 'Browse customers and their details',
                'tables_accessed': ['Customers'],
                'frequency': 0.20,
                'selectivity': 0.05,
                'complexity': 'Low',
                'avg_result_size': 50
            },
            {
                'query_type': 'JOIN',
                'description': 'Fetch customer orders with product details',
                'tables_accessed': ['Customers', 'Orders', 'Products'],
                'frequency': 0.25,
                'selectivity': 0.08,
                'complexity': 'Medium',
                'avg_result_size': 200
            },
            {
                'query_type': 'JOIN',
                'description': 'Inventory and warehouse status lookup',
                'tables_accessed': ['Products', 'Inventory'],
                'frequency': 0.15,
                'selectivity': 0.10,
                'complexity': 'Low',
                'avg_result_size': 100
            },
            {
                'query_type': 'AGGREGATE',
                'description': 'Revenue analytics and sales trends',
                'tables_accessed': ['Orders', 'Payments'],
                'frequency': 0.20,
                'selectivity': 0.50,
                'complexity': 'High',
                'avg_result_size': 1000
            },
            {
                'query_type': 'UPDATE',
                'description': 'Order status updates',
                'tables_accessed': ['Orders'],
                'frequency': 0.10,
                'selectivity': 0.02,
                'complexity': 'Low',
                'avg_result_size': 1
            },
            {
                'query_type': 'INSERT',
                'description': 'New customer registrations and orders',
                'tables_accessed': ['Customers', 'Orders', 'Payments'],
                'frequency': 0.10,
                'selectivity': 0.01,
                'complexity': 'Medium',
                'avg_result_size': 1
            }
        ],
        
        "Healthcare": [
            {
                'query_type': 'SELECT',
                'description': 'Patient demographics and history lookup',
                'tables_accessed': ['Patients'],
                'frequency': 0.18,
                'selectivity': 0.05,
                'complexity': 'Low',
                'avg_result_size': 30
            },
            {
                'query_type': 'JOIN',
                'description': 'Fetch patient appointments and doctor details',
                'tables_accessed': ['Patients', 'Doctors', 'Appointments'],
                'frequency': 0.22,
                'selectivity': 0.08,
                'complexity': 'Medium',
                'avg_result_size': 150
            },
            {
                'query_type': 'JOIN',
                'description': 'Medical records and prescription history',
                'tables_accessed': ['Patients', 'Medical_Records', 'Prescriptions'],
                'frequency': 0.20,
                'selectivity': 0.10,
                'complexity': 'High',
                'avg_result_size': 300
            },
            {
                'query_type': 'AGGREGATE',
                'description': 'Hospital analytics and patient statistics',
                'tables_accessed': ['Appointments', 'Patients', 'Doctors'],
                'frequency': 0.15,
                'selectivity': 0.60,
                'complexity': 'High',
                'avg_result_size': 2000
            },
            {
                'query_type': 'UPDATE',
                'description': 'Appointment status and prescription updates',
                'tables_accessed': ['Appointments', 'Prescriptions'],
                'frequency': 0.12,
                'selectivity': 0.03,
                'complexity': 'Low',
                'avg_result_size': 1
            },
            {
                'query_type': 'INSERT',
                'description': 'New medical records and prescriptions',
                'tables_accessed': ['Medical_Records', 'Prescriptions'],
                'frequency': 0.13,
                'selectivity': 0.01,
                'complexity': 'Medium',
                'avg_result_size': 1
            }
        ],
        
        "University": [
            {
                'query_type': 'SELECT',
                'description': 'Student profile and enrollment details',
                'tables_accessed': ['Students'],
                'frequency': 0.17,
                'selectivity': 0.05,
                'complexity': 'Low',
                'avg_result_size': 40
            },
            {
                'query_type': 'JOIN',
                'description': 'Student enrollments with course details',
                'tables_accessed': ['Students', 'Enrollments', 'Courses'],
                'frequency': 0.21,
                'selectivity': 0.08,
                'complexity': 'Medium',
                'avg_result_size': 180
            },
            {
                'query_type': 'JOIN',
                'description': 'Faculty course assignments and info',
                'tables_accessed': ['Faculty', 'Courses', 'Departments'],
                'frequency': 0.12,
                'selectivity': 0.07,
                'complexity': 'Low',
                'avg_result_size': 80
            },
            {
                'query_type': 'AGGREGATE',
                'description': 'Academic analytics (GPA, enrollment trends)',
                'tables_accessed': ['Grades', 'Enrollments', 'Students'],
                'frequency': 0.18,
                'selectivity': 0.55,
                'complexity': 'High',
                'avg_result_size': 1500
            },
            {
                'query_type': 'UPDATE',
                'description': 'Grade submissions and enrollment status',
                'tables_accessed': ['Grades', 'Enrollments'],
                'frequency': 0.16,
                'selectivity': 0.04,
                'complexity': 'Medium',
                'avg_result_size': 1
            },
            {
                'query_type': 'INSERT',
                'description': 'New student registrations and enrollments',
                'tables_accessed': ['Students', 'Enrollments'],
                'frequency': 0.16,
                'selectivity': 0.02,
                'complexity': 'Low',
                'avg_result_size': 1
            }
        ]
    }
    
    @staticmethod
    def load_schema(schema_name: str) -> Dict[str, Any]:
        """
        Load a predefined schema.
        
        Args:
            schema_name: Name of schema to load
        
        Returns:
            Schema dictionary
        """
        if schema_name not in DataLoader.SCHEMAS:
            raise ValueError(f"Schema '{schema_name}' not found")
        
        return DataLoader.SCHEMAS[schema_name]
    
    @staticmethod
    def get_schema_names() -> List[str]:
        """Get list of available schema names."""
        return list(DataLoader.SCHEMAS.keys())
    
    @staticmethod
    def generate_workload(workload_type: str) -> Dict[str, Any]:
        """
        Generate workload characteristics.
        
        Args:
            workload_type: Type of workload (OLTP, OLAP, Mixed)
        
        Returns:
            Workload characteristics
        """
        
        if workload_type == "OLTP (Transaction-heavy)":
            return {
                'type': 'OLTP',
                'read_ratio': 0.85,
                'write_ratio': 0.15,
                'avg_selectivity': 0.05,
                'query_complexity': 'low',
                'avg_result_size': 100,
                'transaction_size': 'small'
            }
        
        elif workload_type == "OLAP (Analytics-heavy)":
            return {
                'type': 'OLAP',
                'read_ratio': 0.98,
                'write_ratio': 0.02,
                'avg_selectivity': 0.50,
                'query_complexity': 'high',
                'avg_result_size': 10000,
                'transaction_size': 'large'
            }
        
        else:  # Mixed
            return {
                'type': 'Mixed',
                'read_ratio': 0.75,
                'write_ratio': 0.25,
                'avg_selectivity': 0.20,
                'query_complexity': 'medium',
                'avg_result_size': 1000,
                'transaction_size': 'medium'
            }
    
    @staticmethod
    def generate_constraints(num_sites: int, 
                           schema: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate optimization constraints.
        
        Args:
            num_sites: Number of sites
            schema: Database schema
        
        Returns:
            Constraints dictionary
        """
        
        total_tables = len(schema.get('tables', []))
        
        return {
            'storage_per_site': 1000 * num_sites,  # GB per site
            'total_storage': 1000 * num_sites,  # GB total
            'network_bandwidth': 100 * num_sites,  # Mbps per link
            'max_replication_factor': 3,
            'consistency_model': 'ACID',
            'max_fragments_per_table': max(2, num_sites)
        }
    
    @staticmethod
    def get_query_patterns(schema_name: str) -> List[Dict[str, Any]]:
        """
        Get schema-specific query patterns.
        
        Args:
            schema_name: Name of the schema
        
        Returns:
            List of query pattern dictionaries for the schema
        """
        
        if schema_name not in DataLoader.QUERY_PATTERNS:
            raise ValueError(f"Query patterns not found for schema '{schema_name}'")
        
        return DataLoader.QUERY_PATTERNS[schema_name]
    
    @staticmethod
    def get_sample_query_patterns() -> List[Dict[str, Any]]:
        """
        Get default sample query patterns (E-Commerce).
        
        Deprecated: Use get_query_patterns(schema_name) instead
        
        Returns:
            List of query patterns
        """
        return DataLoader.QUERY_PATTERNS.get("E-Commerce", [])
    
    @staticmethod
    def get_query_pattern_summary(schema_name: str) -> Dict[str, Any]:
        """
        Get summary statistics of query patterns for a schema.
        
        Args:
            schema_name: Name of the schema
        
        Returns:
            Summary dictionary with statistics
        """
        
        patterns = DataLoader.get_query_patterns(schema_name)
        
        if not patterns:
            return {}
        
        # Calculate statistics
        avg_complexity = {
            'Low': 0,
            'Medium': 0,
            'High': 0
        }
        
        total_frequency = 0
        total_selectivity = 0
        query_types = set()
        
        for pattern in patterns:
            complexity = pattern.get('complexity', 'Medium')
            avg_complexity[complexity] += 1
            total_frequency += pattern.get('frequency', 0)
            total_selectivity += pattern.get('selectivity', 0)
            query_types.add(pattern.get('query_type', 'UNKNOWN'))
        
        return {
            'schema': schema_name,
            'total_patterns': len(patterns),
            'query_types': list(query_types),
            'complexity_distribution': avg_complexity,
            'avg_frequency': total_frequency / len(patterns) if patterns else 0,
            'avg_selectivity': total_selectivity / len(patterns) if patterns else 0
        }
    
    @staticmethod
    def get_dominant_access_patterns(schema_name: str) -> List[str]:
        """
        Get dominant access patterns (top 3 by frequency) for a schema.
        
        Args:
            schema_name: Name of the schema
        
        Returns:
            List of descriptions of top patterns
        """
        
        patterns = DataLoader.get_query_patterns(schema_name)
        
        # Sort by frequency
        sorted_patterns = sorted(patterns, key=lambda x: x.get('frequency', 0), reverse=True)
        
        # Return top 3 descriptions
        return [p.get('description', 'Unknown') for p in sorted_patterns[:3]]