# Force Field Provenance

 - `lj-sage-2-2-msm-0-expanded-torsions.offxml`
    - Initial force field: `openff-2.2.0.offxml`
    - Steps:
        - Set bonds and angle terms by means from MSM (e.g. https://github.com/jthorton/MSM_QCArchive/blob/master/Mod_sem.ipynb)
        - Set dihedral Ks to 0
        - Expand dihedral periodicities through 1-4

- `lj-sage-2-2-msm-0-expanded-torsions-urey-bradley.offxml`
    - Initial force field: `lj-sage-2-2-msm-0-expanded-torsions.offxml`
    - Steps:
        - Add Urey-Bradley terms to the force field with: `python add_urey_bradley_terms.py lj-sage-2-2-msm-0-expanded-torsions.offxml lj-sage-2-2-msm-0-expanded-torsions-urey-bradley.offxml`

