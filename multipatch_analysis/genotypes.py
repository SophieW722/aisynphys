from __future__ import print_function
import sys
from .constants import GENOTYPES, REPORTER_LINES, DRIVER_LINES, FLUOROPHORES


class Genotype(object):
    """
    Class used to provide phenotype information from a genotype string.

    Parameters
    ----------
    gtype : str
        A genotype string like
        'Tlx3-Cre_PL56/wt;Sst-IRES-FlpO/wt;Ai65F/wt;Ai140(TIT2L-GFP-ICL-tTA2)/wt'

    Notes
    -----

    In the example given above, the genotype specifies two driver lines (Tlx3-Cre_PL65
    and Sst-IRES-FlpO) and two reporter lines (Ai65F and Ai140). This results in
    tdTomato expression in sst cells and GFP expression in tlx3 cells.

    Example usage:

        gtype_str = 'Tlx3-Cre_PL56/wt;Sst-IRES-FlpO/wt;Ai65F/wt;Ai140(TIT2L-GFP-ICL-tTA2)/wt'
        gtype = Genotype(gtype_str)

        gtype.drivers()          # =>  ['tlx3', 'sst']
        gtype.reporters()        # =>  ['tdTomato', 'GFP']
        gtype.reporters('tlx3')  # =>  ['tdTomato'] 

        gtype.colors()           # =>  ['red', 'green']
        gtype.colors('sst')      # =>  ['green']

        gtype.driver_lines()     # =>  ['Tlx3-Cre_PL65', 'Sst-IRES-FlpO']
        gtype.reporter_lines()   # =>  ['Ai65F', 'Ai140']

    """
    def __init__(self, gtype):
        self.gtype = gtype
        self._parse()

    def __repr__(self):
        return "<Genotype %s>" % self.gtype
    
    def drivers(self, reporter=None, colors=None):
        """Return a list of drivers in this genotype (such as pvalb, sst, etc.)

        Parameters
        ----------
        reporter : str or None
            If specified, then only the drivers linked to this reporter are returned.
            Otherwise, all drivers are returned.
        colors : list or None
            If specified, a list of colors that are detected together. The return
            value contains any drivers that could have produced all of the detected
            colors.
        """
        if reporter is None:
            assert colors is None
            return list(self._driver_reporter_map.keys())
        elif colors is not None:
            drivers = []
            for d in self.drivers():
                d_colors = self.colors(driver=d)
                if sorted(d_colors) == sorted(colors):
                    drivers.append(d)
        else:
            return self._reporter_driver_map[reporter]
    
    def reporters(self, driver=None):
        """Return a list of reporters in this genotype (such as GFP, tdTomato, etc.)

        Parameters
        ----------
        driver : str or None
            If specified, then only the reporters linked to this driver are returned.
            Otherwise, all reporters are returned.
        """
        if driver is None:
            return list(self._reporter_driver_map.keys())
        else:
            return self._driver_reporter_map[driver]

    def colors(self, driver=None):
        """Return a list of fluorescent emission colors generated by reporters
        in this genotype.

        Parameters
        ----------
        driver : str or None
            If specified, then only the colors linked to this driver are returned.
            Otherwise, all colors are returned.
        """
        reporters = self.reporters(driver)
        return list(set([FLUOROPHORES[r] for r in reporters]))

    def predict_driver_expression(self, colors):
        """Given information about fluorescent colors expressed in a cell,
        return predictions about whether each driver could have been active.

        Parameters
        ----------
        colors : dict
            Describes whether each color observed in a cell was expressed (True),
            not expressed (False), or ambiguous (None).

        Returns
        -------
        driver_expression : dict
            Dict indicating whether each driver in the genotype could be expressed
            (True), could not be expressed (False), or has no information (None).

        Notes
        -----

        Example colors dict::

            colors = {
                'red':   True,   # cell is red
                'green': False,  # cell is not green
                'blue':  None,   # cell may or may not be blue (no information, or ambiguous appearance)
            }
        """
        drivers = {}
        for driver in self.drivers():
            driver_active = None  # start with no information
            for dcolor in self.colors(driver):
                color_expressed = colors.get(dcolor, None)
                if color_expressed is True and driver_active is not False:
                    driver_active = True
                if color_expressed is False:
                    driver_active = False
            drivers[driver] = driver_active
        return drivers

    def _parse(self):
        """Attempt to predict phenotype information from a genotype string
        """
        ignore = ['wt', 'PhiC31-neo']
        
        parts = set()
        for part in self.gtype.split(';'):
            for subpart in part.split('/'):
                if subpart in ignore:
                    continue
                parts.add(subpart)
                
        self.driver_lines = [p for p in parts if p in DRIVER_LINES]
        self.reporter_lines = [p for p in parts if p in REPORTER_LINES]
        extra = parts - set(self.driver_lines + self.reporter_lines)
        if len(extra) > 0:
            raise Exception("Unknown genotype part(s): %s" % str(extra))
        
        # map {cre line : fluorophore : color} and back again
        reporter_driver_map = {}
        driver_reporter_map = {}

        # Automatically determine driver-reporter mapping.
        try:
            for d in self.driver_lines:
                driver_factors, driver = DRIVER_LINES[d]
                # driver_factors is a list containing one or more of 'cre', 'flp', and 'tTA'
                # (these are _provided_ by the driver)
                for r in self.reporter_lines:
                    reporter_factors, reporters = REPORTER_LINES[r]
                    # reporter_factors is an '&'-delimited string with one or more 'cre', 'flp', and 'tTA'
                    # (these are _required_ by the reporter)
                    reporter_factors = reporter_factors.split('&')
                    if not all([rf in driver_factors for rf in reporter_factors]):
                        # this is an oversimplification, but should work for now..
                        continue
                    driver_reporter_map.setdefault(driver, []).extend(reporters)
                    for r in reporters:
                        reporter_driver_map.setdefault(r, []).append(driver)
            parse_ok = True
        except Exception as parse_exc:
            sys.excepthook(*sys.exc_info())
            parse_ok = False

        # For now, we don't really trust the above automated method.
        # Try to pull the same information from a list of known genotypes,
        # and if it's not there, then we can give an error message with
        # the suggested mapping.
        if GENOTYPES.get(self.gtype) is None:
            if parse_ok:
                raise Exception('Unknown genotype "%s".\nSuggested addition to constants.GENOTYPES:\n    %r\n.' % 
                                (self.gtype, driver_reporter_map))
            else:
                raise Exception('Unknown genotype "%s" (and genotype parsing failed).' % self.gtype)

        drm = GENOTYPES[self.gtype]
        self._driver_reporter_map = drm
        # generate the reverse mapping
        self._reporter_driver_map = {}
        for d,rs in drm.items():
            for r in rs:
                self._reporter_driver_map.setdefault(r, []).append(driver)


if __name__ == '__main__':
    for gt in GENOTYPES:
        print(gt)
        g = Genotype(gt)
        print("    " + "   ".join(["%s: %s" % (d, ','.join(g.colors(d))) for d in g.drivers()]))
        print("")
    print(g.predict_driver_expression({'green': True, 'red': False}))
