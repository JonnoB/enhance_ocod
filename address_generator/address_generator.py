import yaml
import os
import random
from typing import Dict, List, Any


class AddressGenerator:
    def __init__(self, config_path: str):
        self.config_path = config_path  
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Load imported name files
        self._load_imports()
        
        self.component_builder = ComponentBuilder(self.config)
        self.structure_generator = StructureGenerator(self.config)
        # self.corruption_engine = CorruptionEngine(self.config)
        
    def _load_imports(self):
        """Load external name files specified in imports section"""
        if 'imports' in self.config:
            config_dir = os.path.dirname(self.config_path) if hasattr(self, 'config_path') else '.'
            
            for key, import_path in self.config['imports'].items():
                file_path, yaml_key = import_path.split(':')
                full_path = os.path.join(config_dir, file_path)
                
                try:
                    with open(full_path, 'r') as f:
                        import_data = yaml.safe_load(f)
                    
                    # Check if the key exists in the imported data
                    if import_data is None:
                        print(f"Warning: {file_path} is empty or invalid")
                        continue
                        
                    if yaml_key not in import_data:
                        print(f"Warning: Key '{yaml_key}' not found in {file_path}")
                        continue
                    
                    # Add to components section
                    if 'components' not in self.config:
                        self.config['components'] = {}
                    self.config['components'][key] = import_data[yaml_key]
                    
                except FileNotFoundError:
                    print(f"Warning: Import file {full_path} not found, skipping {key}")
                    continue
                except Exception as e:
                    print(f"Error loading {file_path}: {e}")
                    continue

    def _format_address(self, components: Dict) -> Dict[str, Any]:
        """Build the final address string and create spans with positions"""
        address_parts = []
        spans = []
        current_pos = 0
        
        # Define the typical order of address components
        component_order = [
            'unit_type', 'unit_id', 'building_name', 'street_number', 
            'street_name', 'city', 'postcode'
        ]
        
        for component_type in component_order:
            if component_type in components and components[component_type]:
                component_text = str(components[component_type])
                
                # Add separator if not the first component
                if address_parts:
                    separator = ", "
                    address_parts.append(separator)
                    current_pos += len(separator)
                
                # Add the component
                start_pos = current_pos
                address_parts.append(component_text)
                end_pos = current_pos + len(component_text)
                
                # Create span
                spans.append({
                    'text': component_text,
                    'start': start_pos,
                    'end': end_pos,
                    'label': component_type
                })
                
                # Update current position
                current_pos = end_pos
        
        # Join all parts to create final address
        full_address = ''.join(address_parts)
        
        return {
            'text': full_address,
            'spans': spans
        }

    def generate_address(self) -> Dict[str, Any]:
        # Select property type
        property_type = self._sample_property_type()
        
        # Generate structure (single/range/list)
        structure = self.structure_generator.generate_structure(property_type)
        
        # Build components
        address_components = self.component_builder.build_components(structure)
        
        # Apply corruption (commented out for now since CorruptionEngine is disabled)
        # corrupted_components = self.corruption_engine.apply_corruption(address_components)
        
        # Format into final address text and spans
        return self._format_address(address_components)

    def _sample_property_type(self) -> str:
        types = list(self.config['parameters']['property_types'].keys())
        weights = list(self.config['parameters']['property_types'].values())
        return random.choices(types, weights=weights)[0]
    
class StructureGenerator:
    def __init__(self, config: Dict):
        self.config = config
    
    def generate_structure(self, property_type: str) -> Dict:
        # Decide if this is single property, range, or list
        structure_type = self._choose_structure_type()
        
        return {
            'property_type': property_type,
            'structure_type': structure_type,
            'num_properties': self._decide_num_properties(structure_type),
            'has_nesting': self._decide_nesting(property_type),
            'multi_building': self._decide_multi_building(property_type)
        }
    
    def _choose_structure_type(self) -> str:
        choices = ['single', 'range', 'list']
        probs = [
            self.config['parameters']['structure']['single_probability'],
            self.config['parameters']['structure']['range_probability'],
            self.config['parameters']['structure']['list_probability']
        ]
        return random.choices(choices, weights=probs)[0]
    
    def _decide_num_properties(self, structure_type: str) -> int:
        """Decide how many properties in this address"""
        if structure_type == 'single':
            return 1
        elif structure_type == 'range':
            return random.randint(2, 10)  # Range like "5 to 15"
        else:  # list
            return random.randint(2, 5)   # List like "1, 3, and 5"
    
    def _decide_nesting(self, property_type: str) -> bool:
        """Some property types more likely to have nested structures"""
        nesting_probs = {
            'residential': 0.3,
            'business': 0.2, 
            'land': 0.1,
            'carpark': 0.4,
            'airspace': 0.1
        }
        return random.random() < nesting_probs.get(property_type, 0.2)
    
    def _decide_multi_building(self, property_type: str) -> bool:
        """Whether this address spans multiple buildings/roads"""
        multi_probs = {
            'residential': 0.1,
            'business': 0.3,
            'land': 0.2, 
            'carpark': 0.1,
            'airspace': 0.05
        }
        return random.random() < multi_probs.get(property_type, 0.1)

class ComponentBuilder:
    def __init__(self, config: Dict):
        self.config = config
    
    def _add_possessive(self, name: str) -> str:
        """Maybe add possessive 's to a name"""
        if random.random() < self.config['parameters']['name_generation']['possessive_probability']:
            # Handle names ending in 's' differently
            if name.endswith('s'):
                return f"{name}'"
            else:
                return f"{name}'s"
        return name
    
    def generate_street_name(self) -> str:
        """Generate street name with possible possessive"""
        use_base = random.random() < self.config['parameters']['street_generation']['use_base_name_probability']
        
        if use_base:
            base_name = random.choice(self.config['components']['base_names'])
            base_name = self._add_possessive(base_name)
            
            if random.random() < self.config['parameters']['street_generation']['suffix_probability']:
                suffix = random.choice(self.config['components']['street_suffixes'])
                return f"{base_name} {suffix}"
            else:
                return base_name
        else:
            standalone = random.choice(self.config['components']['standalone_streets'])
            return self._add_possessive(standalone)
    
    def generate_building_name(self, property_type: str) -> str:
        """Generate building name with possible possessive"""
        base_name = random.choice(self.config['components']['base_names'])
        base_name = self._add_possessive(base_name)
        
        if property_type == 'residential':
            suffixes = self.config['components']['building_suffixes']['residential']
        else:
            suffixes = self.config['components']['building_suffixes']['commercial']
        
        if random.random() < self.config['parameters']['building_generation']['suffix_probability']:
            suffix = random.choice(suffixes)
            return f"{base_name} {suffix}"
        else:
            return base_name

    def build_components(self, structure: Dict) -> Dict:
        """Build all address components based on structure"""
        components = {}
        property_type = structure['property_type']
        
        # Generate street components
        components['street_name'] = self.generate_street_name()
        
        # Generate street number (could be single, range, or list based on structure)
        if structure['structure_type'] == 'range':
            start = random.randint(1, 50)
            end = start + random.randint(2, 20)
            filter_type = random.choice(['odds', 'evens', None])
            components['street_number'] = self.generate_number_range(start, end, filter_type)
            if filter_type:
                components['number_filter'] = filter_type
        elif structure['structure_type'] == 'list':
            numbers = sorted(random.sample(range(1, 100), random.randint(2, 5)))
            components['street_number'] = self.generate_number_list(numbers)
        else:
            components['street_number'] = str(random.randint(1, 999))
        
        # Add building name sometimes
        if random.random() < 0.3:  # 30% chance of building name
            components['building_name'] = self.generate_building_name(property_type)
        
        # Add unit info for flats/apartments
        if property_type == 'residential' and random.random() < 0.4:
            components['unit_type'] = random.choice(self.config['components']['unit_types'])
            components['unit_id'] = str(random.randint(1, 50))
        
        # Add city and postcode
        components['city'] = random.choice(self.config['components']['cities'])
        components['postcode'] = self._generate_postcode()
        
        return components

    def _generate_postcode(self) -> str:
        """Generate a realistic UK postcode with upper, lower, or mixed case"""
        import random
        
        # UK postcode areas (first 1-2 letters)
        areas = [
            'B', 'BA', 'BB', 'BD', 'BH', 'BL', 'BN', 'BR', 'BS', 'BT',
            'CA', 'CB', 'CF', 'CH', 'CM', 'CO', 'CR', 'CT', 'CV', 'CW',
            'DA', 'DD', 'DE', 'DG', 'DH', 'DL', 'DN', 'DT', 'DY',
            'E', 'EC', 'EH', 'EN', 'EX',
            'FK', 'FY',
            'G', 'GL', 'GU',
            'HA', 'HD', 'HG', 'HP', 'HR', 'HS', 'HU', 'HX',
            'IG', 'IP', 'IV',
            'KA', 'KT', 'KW', 'KY',
            'L', 'LA', 'LD', 'LE', 'LL', 'LN', 'LS', 'LU',
            'M', 'ME', 'MK', 'ML',
            'N', 'NE', 'NG', 'NN', 'NP', 'NR', 'NW',
            'OL', 'OX',
            'PA', 'PE', 'PH', 'PL', 'PO', 'PR',
            'RG', 'RH', 'RM',
            'S', 'SA', 'SE', 'SG', 'SK', 'SL', 'SM', 'SN', 'SO', 'SP', 'SR', 'SS', 'ST', 'SW', 'SY',
            'TA', 'TD', 'TF', 'TN', 'TQ', 'TR', 'TS', 'TW',
            'UB',
            'W', 'WA', 'WC', 'WD', 'WF', 'WN', 'WR', 'WS', 'WV',
            'YO'
        ]
        
        # Valid letters for district (avoiding I, Q, Z in certain positions)
        district_letters = 'ABCDEFGHJKLMNOPRSTUWXY'
        
        # Valid letters for final two characters (avoiding C, I, K, M, O, V)
        final_letters = 'ABDEFGHJLNPQRSTUWXYZ'
        
        def apply_case_style(text, style):
            """Apply case style to text"""
            if style == 'upper':
                return text.upper()
            elif style == 'lower':
                return text.lower()
            else:  # mixed - horrible mixture
                return ''.join(char.lower() if random.random() < 0.5 else char.upper() 
                            for char in text)
        
        # Choose case style: 45% upper, 45% lower, 10% horrible mixture
        case_style = random.choices(
            ['upper', 'lower', 'mixed'], 
            weights=[0.45, 0.45, 0.10]
        )[0]
        
        # Choose area code
        area = apply_case_style(random.choice(areas), case_style)
        
        # District number (1-99, but typically 1-20 for most areas)
        district_num = random.randint(1, 20)
        
        # Sometimes add a letter after the district number
        district = str(district_num)
        if random.random() < 0.3:  # 30% chance of having a letter
            district += apply_case_style(random.choice(district_letters), case_style)
        
        # Sector number (0-9)
        sector = random.randint(0, 9)
        
        # Unit (2 letters)
        unit = apply_case_style(''.join(random.choices(final_letters, k=2)), case_style)
        
        # Sometimes include space, sometimes don't
        space = ' ' if random.random() < 0.8 else ''
        
        return f"{area}{district}{space}{sector}{unit}"

    def generate_number_range(self, start: int, end: int, filter_type: str = None, 
                            connector_style: str = "to", spacing: str = "normal", 
                            parentheses: bool = True) -> str:
        # Choose connector
        if connector_style == "to":
            connector = " to "
        elif connector_style == "dash":
            connector = "-" if spacing == "tight" else " - "
        
        # Build range part
        range_part = f"{start}{connector}{end}"
        
        # Add filter if present
        if filter_type:
            if parentheses:
                filter_part = f" ({filter_type} only)" if filter_type in ["odds", "evens"] else f" ({filter_type})"
            else:
                filter_part = f" {filter_type}"
            return range_part + filter_part
        
        return range_part

    def generate_number_list(self, numbers: List[int], final_connector: str = "and") -> str:
        if len(numbers) == 1:
            return str(numbers[0])
        elif len(numbers) == 2:
            return f"{numbers[0]} {final_connector} {numbers[1]}"
        else:
            return f"{', '.join(map(str, numbers[:-1]))}, {final_connector} {numbers[-1]}"
    