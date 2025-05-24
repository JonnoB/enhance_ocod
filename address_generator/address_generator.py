import yaml
import os
import random
from typing import Dict, List, Any
import re
from typing import List, Tuple


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
        
        # Generate structure
        structure = self.structure_generator.generate_structure(property_type)
        
        # Build tagged components
        tagged_components = self.component_builder.build_components(structure)
        
        # Apply corruption (temporarily skip until implemented)
        # corrupted_components = self.corruption_engine.apply_corruption(tagged_components)
        corrupted_components = tagged_components  # Direct assignment for now
        
        # Parse tags and format final address
        return self._parse_tags_and_format(corrupted_components)

    def _parse_tags_and_format(self, components: Dict) -> Dict[str, Any]:
        """Parse tags from components and create final address with spans"""
        
        # Build raw address with tags
        component_order = [
            'unit_type', 'unit_id', 'building_name', 'street_number', 
            'street_name', 'city', 'postcode'
        ]
        
        tagged_parts = []
        for component_type in component_order:
            if component_type in components and components[component_type]:
                tagged_parts.append(str(components[component_type]))
        
        # Join with commas
        tagged_address = ", ".join(tagged_parts)
        
        # Parse tags and create spans
        return self._extract_spans_from_tags(tagged_address)

    def _extract_spans_from_tags(self, tagged_text: str) -> Dict[str, Any]:
        """Extract entities from tagged text and create clean text + spans"""
        
        # Updated pattern to handle underscores in tag names
        tag_pattern = r'<([\w_]+)>(.*?)</\1>'
        matches = list(re.finditer(tag_pattern, tagged_text))
        
        spans = []
        clean_text = tagged_text
        
        # Process matches from left to right, tracking cumulative offset
        cumulative_offset = 0
        
        for match in matches:
            tag_name = match.group(1)
            content = match.group(2)
            
            # Position in original tagged text
            tagged_start = match.start()
            tagged_end = match.end()
            
            # Position in clean text (accounting for previously removed tags)
            clean_start = tagged_start - cumulative_offset
            clean_end = clean_start + len(content)
            
            # Create span
            spans.append({
                'text': content,
                'start': clean_start,
                'end': clean_end,
                'label': tag_name
            })
            
            # Update cumulative offset
            opening_tag = f"<{tag_name}>"
            closing_tag = f"</{tag_name}>"
            tag_overhead = len(opening_tag) + len(closing_tag)
            cumulative_offset += tag_overhead
        
        # Remove all tags to create clean text
        clean_text = re.sub(tag_pattern, r'\2', tagged_text)
        
        return {
            'text': clean_text,
            'spans': spans
        }

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
                return f"<street_name>{base_name} {suffix}</street_name>"
            else:
                return f"<street_name>{base_name}</street_name>"
        else:
            standalone = random.choice(self.config['components']['standalone_streets'])
            return f"<street_name>{self._add_possessive(standalone)}</street_name>"
    
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
            return f"<building_name>{base_name} {suffix}</building_name>"
        else:
            return f"<building_name>{base_name}</building_name>"

    def build_components(self, structure: Dict) -> Dict:
        """Build tagged address components"""
        components = {}
        property_type = structure['property_type']
        
        # Generate tagged components
        if structure['structure_type'] == 'range':
            start = random.randint(1, 50)
            end = start + random.randint(2, 20)
            filter_type = random.choice(['odds', 'evens', None])
            components['street_number'] = self.generate_number_range(start, end, filter_type)
        elif structure['structure_type'] == 'list':
            numbers = sorted(random.sample(range(1, 100), random.randint(2, 5)))
            components['street_number'] = self.generate_number_list(numbers)
        else:
            components['street_number'] = f"<street_number>{random.randint(1, 999)}</street_number>"
        
        # Other tagged components
        components['street_name'] = self.generate_street_name()
        
        if random.random() < 0.3:
            components['building_name'] = self.generate_building_name(property_type)
        
        if property_type == 'residential' and random.random() < 0.4:
            unit_type = random.choice(self.config['components']['unit_types'])
            unit_id = str(random.randint(1, 50))
            components['unit_type'] = f"<unit_type>{unit_type}</unit_type>"
            components['unit_id'] = f"<unit_id>{unit_id}</unit_id>"
        
        components['city'] = f"<city>{random.choice(self.config['components']['cities'])}</city>"
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
        
        return f"<postcode>{area}{district}{space}{sector}{unit}</postcode>"

    def generate_number_range(self, start: int, end: int, filter_type: str = None, 
                            connector_style: str = "to", spacing: str = "normal", 
                            parentheses: bool = True, plural_probability: float = 0.5) -> str:
        # Build the range text
        if connector_style == "to":
            connector = " to "
        elif connector_style == "dash":
            connector = "-" if spacing == "tight" else " - "
        
        range_part = f"{start}{connector}{end}"
        
        if filter_type:
            # Handle optional plurals for odds/evens
            if filter_type in ["odds", "evens"]:
                if random.random() < plural_probability:
                    filter_text = filter_type  # "odds" or "evens" 
                else:
                    filter_text = filter_type[:-1]  # "odd" or "even"
            else:
                filter_text = filter_type  # other filter types unchanged
            
            if parentheses:
                return f"<street_number>{range_part}</street_number> (<number_filter>{filter_text}</number_filter> only)"
            else:
                return f"<street_number>{range_part}</street_number> <number_filter>{filter_text}</number_filter>"
        else:
            return f"<street_number>{range_part}</street_number>"

    def generate_number_list(self, numbers: List[int], final_connector: str = "and") -> str:
        """Generate individual tagged numbers in a list"""
        if len(numbers) == 1:
            return f"<street_number>{numbers[0]}</street_number>"
        elif len(numbers) == 2:
            return f"<street_number>{numbers[0]}</street_number> {final_connector} <street_number>{numbers[1]}</street_number>"
        else:
            # For longer lists: "2, 4, and 6"
            tagged_numbers = [f"<street_number>{num}</street_number>" for num in numbers[:-1]]
            result = ", ".join(tagged_numbers)
            result += f", {final_connector} <street_number>{numbers[-1]}</street_number>"
            return result

