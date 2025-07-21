# -*- coding: utf-8 -*-
"""
@author: EHSAN_ab

"""
import numpy as np
import pandas as pd
from pymatgen.core import Composition
from pymatgen.core.periodic_table import Element
import os

class SuperconductorDescriptorGenerator:
    """Generates physics-informed descriptors and elemental property features for superconductors"""
    def __init__(self):
        # Path to cleaned elemental properties table
        ptable_path = r"path to ptable o elements"
        self.element_df = pd.read_csv(ptable_path)
        self.element_df.set_index('symbol', inplace=True)
        
        # Properties used in physics-informed descriptors
        self.physics_props = ['mag', 'lambda', 'en', 'ar', 'mp', 've', 'Tc']
        
        # Get all numeric columns for additional features
        self.all_numeric_props = self.element_df.select_dtypes(include=np.number).columns.tolist()
        self.additional_props = [prop for prop in self.all_numeric_props 
                                 if prop not in self.physics_props]
        
        # Create default values dictionary
        self.default_physics = {
            'mag': 0.0, 
            'lambda': 0.3,  # Eliashberg default
            'en': 1.5, 
            'ar': 1.5,   
            'mp': 1000, 
            've': 0,
            'Tc': 0  # Default for missing Tc
        }
        
        # Create default values for additional properties (mean imputation)
        self.default_additional = self.element_df[self.additional_props].mean().to_dict()
        
        # Transition metals list
        self.transition_metals = [
            'Sc','Ti','V','Cr','Mn','Fe','Co','Ni','Cu','Zn',
            'Y','Zr','Nb','Mo','Tc','Ru','Rh','Pd','Ag','Cd',
            'Hf','Ta','W','Re','Os','Ir','Pt','Au','Hg'
        ]

    def get_element_property(self, element, prop):
        """Get elemental property with robust NaN handling"""
        try:
            value = self.element_df.loc[element, prop]
            # Handle NaN values
            if pd.isna(value):
                if prop in self.physics_props:
                    return self.default_physics[prop]
                return self.default_additional[prop]
            return value
        except KeyError:
            if prop in self.physics_props:
                return self.default_physics[prop]
            return self.default_additional[prop]
    
    def compute_descriptors(self, formula):
        """Compute physics-informed descriptors and elemental property features from chemical formula"""
        try:
            comp = Composition(formula)
            elements = list(comp.as_dict().keys())
            amounts = list(comp.as_dict().values())
            total_atoms = sum(amounts)
            
            # Calculate atomic fractions
            atomic_fractions = [amt/total_atoms for amt in amounts]
            
            # Get element properties with NaN protection
            physics_properties = {}
            for prop in self.physics_props:
                physics_properties[prop] = [
                    self.get_element_property(e, prop) for e in elements
                ]
            
            # Get additional properties
            additional_properties = {}
            for prop in self.additional_props:
                additional_properties[prop] = [
                    self.get_element_property(e, prop) for e in elements
                ]
            
            # Element fractions
            o_frac = comp.get("O", 0) / total_atoms
            cu_frac = comp.get("Cu", 0) / total_atoms
            fe_frac = comp.get("Fe", 0) / total_atoms
            
            # Atomic numbers
            atomic_numbers = [Element(e).Z for e in elements]
            
            # Valence states with error handling
            valence_states = []
            for e in elements:
                try:
                    valence_states.append(Element(e).common_oxidation_states[0])
                except (IndexError, AttributeError):
                    valence_states.append(0)
            
            # =====================================================
            # DESCRIPTOR CALCULATIONS (Physics-Informed Features)
            # =====================================================
            descriptors = {
                # ========================
                # PRIMARY DESCRIPTORS (7)
                # ========================
                
                # O/Cu ratio: Quantifies hole doping in cuprates
                # Formula: [O]/[Cu]
                "O/Cu_ratio": o_frac / (cu_frac + 1e-6),
                
                # Fe fraction: Absolute iron content
                # Formula: n(Fe)/N
                "Fe_fraction": fe_frac,
                
                # TM fraction: Fraction of transition metals
                # Formula: Σ(n(TMᵢ)/N) for TMᵢ ∈ transition metals
                "TM_fraction": sum(frac for e, frac in zip(elements, atomic_fractions) 
                                   if e in self.transition_metals),
                
                # Fe/(Fe+O) ratio: Distinguishes Fe-Si vs Fe-Co systems
                # Formula: n(Fe)/(n(Fe) + n(O))
                "Fe/(Fe+O)_ratio": fe_frac / (fe_frac + o_frac + 1e-6),
                
                # Magnetic average: Weighted magnetic moment
                # Formula: Σ(fᵢ × μᵢ)
                "Magnetic_avg": sum(
                    mag * frac for mag, frac in zip(physics_properties['mag'], atomic_fractions)
                ),
                
                # Lambda average: Weighted electron-phonon coupling
                # Formula: Σ(fᵢ × λᵢ)
                "Lambda_avg": sum(
                    lam * frac for lam, frac in zip(physics_properties['lambda'], atomic_fractions)
                ),
                
                # Tc average: Weighted superconducting transition temperature
                # Formula: Σ(fᵢ × Tcᵢ)
                "Tc_avg": sum(
                    tc * frac for tc, frac in zip(physics_properties['Tc'], atomic_fractions)
                ),
                
                # ========================
                # CUPARTE-SPECIFIC (6)
                # ========================
                
                # CuO₂ planes: Estimated number of CuO₂ planes
                # Formula: min(n(Cu), 4)
                "CuO2_planes": min(comp.get("Cu", 0), 4),
                
                # Apical O distance: Ba-content dependent
                # Formula: 2.35 + 0.1 × [Ba]
                "Apical_O_dist": 2.35 + 0.1 * comp.get("Ba", 0)/total_atoms,
                
                # Hole doping: Zhang-Rice singlet model
                # Formula: [O]/([O] + 0.5[Cu])
                "Hole_doping": o_frac / (o_frac + 0.5 * cu_frac + 1e-6),
                
                # Jahn-Teller flag: Presence of Mn/Cu
                # Formula: I(Mn ∈ formula ∨ Cu ∈ formula)
                "Jahn_Teller": 1 if ('Mn' in elements or 'Cu' in elements) else 0,
                
                # Perovskite distortion: Deviation from ideal ratio
                # Formula: [O]/[Cu] - 0.33
                "Perovskite_distortion": (comp.get("O", 0)/(comp.get("Cu", 0) + 1e-6)) - 0.33,
                
                # Charge reservoir: Alkaline earth content
                # Formula: [Ba] + [Sr]
                "Charge_reservoir": comp.get("Ba", 0)/total_atoms + comp.get("Sr", 0)/total_atoms,
                
                # ========================
                # IRON-BASED SPECIFIC (7)
                # ========================
                
                # Fe layer separation: Se-content dependent
                # Formula: 5.2 + 0.3 × [Se]
                "Fe_layer_sep": 5.2 + 0.3 * comp.get("Se", 0)/total_atoms,
                
                # Chalcogen height: Te-content dependent
                # Formula: 1.38 + 0.2 × [Te]
                "Chalcogen_height": 1.38 + 0.2 * comp.get("Te", 0)/total_atoms,
                
                # Pnictogen ratio: As/P balance
                # Formula: [As]/([P] + [As])
                "Pnictogen_ratio": comp.get("As", 0)/max(1, comp.get("P", 0) + comp.get("As", 0)),
                
                # Fe-Fe distance: Co-content dependent
                # Formula: 2.8 - 0.1 × [Co]
                "Fe_Fe_distance": 2.8 - 0.1 * comp.get("Co", 0)/total_atoms,
                
                # Magnetic coupling: Fe-Co competition
                # Formula: 0.8[Fe] - 0.2[Co]
                "Magnetic_coupling": 0.8 * fe_frac - 0.2 * comp.get("Co", 0)/total_atoms,
                
                # Structural anisotropy: S/Chalcogen ratio
                # Formula: [S]/([S] + [Se] + [Te])
                "Structural_anisotropy": self._calc_anisotropy(comp),
                
                # Rare earth radius: Weighted atomic radius
                # Formula: Σ(fᵢ × rᵢ) for i ∈ {La, Sm, Nd, Y}
                "Rare_earth_radius": sum(
                    self.get_element_property(e, 'ar') * frac 
                    for e, frac in zip(elements, atomic_fractions) 
                    if e in ['La','Sm','Nd','Y']
                ),
                
                # ========================
                # CONVENTIONAL SC (6)
                # ========================
                
                # Phonon frequency average: B-content dependent
                # Formula: 300 + 100 × [B]
                "Phonon_freq_avg": 300 + 100 * comp.get("B", 0)/total_atoms,
                
                # DOS at Fermi: Nb-content dependent
                # Formula: 0.5 + 0.3 × [Nb]
                "DOS_Fermi": 0.5 + 0.3 * comp.get("Nb", 0)/total_atoms,
                
                # Isotope effect: C-content dependent
                # Formula: 0.4 - 0.1 × [C]
                "Isotope_effect": 0.4 - 0.1 * comp.get("C", 0)/total_atoms,
                
                # BCS gap ratio: Sn-content dependent
                # Formula: 3.5 - 0.5 × [Sn]
                "BCS_gap_ratio": 3.5 - 0.5 * comp.get("Sn", 0)/total_atoms,
                
                # Covalent character: Weighted electronegativity
                # Formula: Σ(fᵢ × ENᵢ)
                "Covalent_character": sum(
                    en * frac for en, frac in zip(physics_properties['en'], atomic_fractions)
                ),
                
                # Electron concentration: Weighted valence electrons
                # Formula: Σ(fᵢ × VEᵢ)
                "Electron_conc": sum(
                    ve * frac for ve, frac in zip(physics_properties['ve'], atomic_fractions)
                ),
                
                # ========================
                # GENERAL DESCRIPTORS (9)
                # ========================
                
                # Atomic size variance: Variance of atomic radii
                # Formula: Var(rᵢ)
                "Atomic_size_var": np.var(physics_properties['ar']) if len(elements) > 1 else 0,
                
                # Melting point average: Weighted melting point
                # Formula: Σ(fᵢ × MPᵢ)
                "Melting_point_avg": sum(
                    mp * frac for mp, frac in zip(physics_properties['mp'], atomic_fractions)
                ),
                
                # Ionicity index: Mean electronegativity deviation
                # Formula: mean(ENᵢ) - 1.5
                "Ionicity_index": np.mean(physics_properties['en']) - 1.5,
                
                # Crystal complexity: Number of unique elements
                # Formula: count(unique elements)
                "Crystal_complexity": len(elements),
                
                # Weighted atomic number: Weighted proton count
                # Formula: Σ(fᵢ × Zᵢ)
                "Weighted_Z": sum(z * frac for z, frac in zip(atomic_numbers, atomic_fractions)),
                
                # Valence imbalance: Absolute sum of oxidation states
                # Formula: |Σ(fᵢ × OSᵢ)|
                "Valence_imbalance": abs(sum(
                    vs * frac for vs, frac in zip(valence_states, atomic_fractions)
                )),
                
                # Lattice stability: Inverse complexity
                # Formula: 1/(1 + N_elements)
                "Lattice_stability": 1 / (1 + len(elements))
            }
            
            # ==========================================
            # ADDITIONAL ELEMENTAL PROPERTY FEATURES
            # ==========================================
            # Compute weighted averages for additional properties
            for prop in self.additional_props:
                prop_values = additional_properties[prop]
                weighted_avg = sum(frac * val for frac, val in zip(atomic_fractions, prop_values))
                descriptors[prop] = weighted_avg
            
            return descriptors
        
        except Exception as e:
            print(f"Error processing {formula}: {str(e)}")
            # Create empty descriptors dictionary with zeros for all features
            all_descriptor_names = self.get_descriptor_names() + self.additional_props
            return {name: 0 for name in all_descriptor_names}
    
    def _calc_anisotropy(self, comp):
        """Calculate structural anisotropy"""
        chalcogens = ['S', 'Se', 'Te']
        s_count = comp.get("S", 0)
        total_chalc = sum(comp.get(ch, 0) for ch in chalcogens)
        return s_count / total_chalc if total_chalc > 0 else 0
    
    def get_descriptor_names(self):
        """Return physics-informed descriptor names"""
        return [
            # Primary descriptors
            "O/Cu_ratio", "Fe_fraction", "TM_fraction", "Fe/(Fe+O)_ratio", 
            "Magnetic_avg", "Lambda_avg", "Tc_avg",
            
            # Cuprate-specific
            "CuO2_planes", "Apical_O_dist", "Hole_doping", "Jahn_Teller", 
            "Perovskite_distortion", "Charge_reservoir",
            
            # Iron-based specific
            "Fe_layer_sep", "Chalcogen_height", "Pnictogen_ratio", "Fe_Fe_distance", 
            "Magnetic_coupling", "Structural_anisotropy", "Rare_earth_radius",
            
            # Conventional SC
            "Phonon_freq_avg", "DOS_Fermi", "Isotope_effect", "BCS_gap_ratio", 
            "Covalent_character", "Electron_conc",
            
            # General descriptors
            "Atomic_size_var", "Melting_point_avg", "Ionicity_index", 
            "Crystal_complexity", "Weighted_Z", "Valence_imbalance", 
            "Lattice_stability"
        ]

# Main execution
if __name__ == "__main__":
    # Initialize descriptor generator
    generator = SuperconductorDescriptorGenerator()
    
    # I/O Paths
    input_csv_path = r"path_to_supecon_dataset or ICSD"
    output_dir = r"output path of supecon_dataset or ICSD"
    output_path = os.path.join(output_dir, "sample_HT_featc2.csv")
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Process data
    df_csv = pd.read_csv(input_csv_path)
    all_data = []
    
    # Preserve these columns from the original CSV
    preserve_columns = ['formula', 'class', 'target']
    
    # Print input columns for verification
    print(f"Input CSV columns: {df_csv.columns.tolist()}")
    
    for idx, row in df_csv.iterrows():
        formula = row['formula']
        desc = generator.compute_descriptors(formula)
        
        # Create result dictionary preserving original columns
        result = {}
        for col in preserve_columns:
            if col in row:
                result[col] = row[col]
            else:
                print(f"Warning: Column '{col}' not found in input row {idx}")
                result[col] = None
        
        result.update(desc)
        all_data.append(result)
    
    # Create and save DataFrame
    df = pd.DataFrame(all_data)
    
    # Reorder columns: preserved columns first, then physics descriptors, then additional features
    preserved_cols = [col for col in preserve_columns if col in df.columns]
    physics_names = generator.get_descriptor_names()
    additional_names = generator.additional_props
    
    column_order = preserved_cols + physics_names + additional_names
    # Only include columns that actually exist in the DataFrame
    column_order = [col for col in column_order if col in df.columns]
    df = df[column_order]
    
    df.to_csv(output_path, index=False)
    
    # Report statistics
    print(f"\nProcessed {len(df)} materials")
    print(f"Generated {len(physics_names)} physics-informed descriptors")
    print(f"Generated {len(additional_names)} additional elemental property features")
    print(f"Total features: {len(physics_names) + len(additional_names)}")
    print(f"Preserved columns: {preserved_cols}")
    print(f"Output saved to: {output_path}")
    print("\nSample output:")
df    