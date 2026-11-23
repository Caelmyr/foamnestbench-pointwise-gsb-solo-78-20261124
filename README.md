# Foam Nest Bench

Check the fit of a padded storage tray from calibrated photographs. The synthetic tray shows a bronze mount, its foam liner and a pressure-transfer card. Register four points around the same flat plane for each photograph. Enter the actual tray dimensions, then mark the object, pressure witness, approach line or other area needed for the measurement.

Use Object outline for a projected silhouette, Lift access for the continuous clear length of a hand approach and Cavity section for bright floor spans. Bearing patches and Bearing change inspect visible transfer color before and after a reseat. Liner grain direction, Rub mark map, Grain-aware rub map, Liner tone, Paired tray visibility, Tray landmark offsets, Label window and Liner bridges help check the condition and space around the mount. Saved readings retain the calibrated overlay, measurements and sources. The image preparation room handles exposure fusion, geometry, region masks and exports before registration.

### Grain-aware rub map

The plain Rub mark map subtracts an isotropic local blur, so a liner cut with diagonal grooves flags the regular groove stripes as scuffs. Grain-aware rub map instead models the groove itself. Mark two texture references (grain A and grain B), one confirmed rub mark and one intact control patch, then set the minimum retained true-scuff area and maximum tolerated false area on intact liner. Each candidate grain direction is tried in turn. The groove brightness is a periodic function of the coordinate perpendicular to that line; a robust phase-binned median builds the groove baseline directly on the calibrated plane, and only pixels darker than that baseline are mark candidates. A groove stripe that is continuously visible along the direction is therefore treated as background, while a scuff that breaks the periodicity survives in the residual. When both directions satisfy the limits, the map with the least false area on intact liner is chosen (a tie is broken by retained scuff area); otherwise the closest candidate is returned flagged as not meeting the limits. Missing-capture pixels never enter the baseline and are painted separately with a safety halo, so two visible scuff halves are never joined into one mark across unseen material.

Use PNG, JPEG or TIFF up to 20 MiB and 14 megapixels. One run may hold 12 photographs and 100 marks. The longest calibrated side is limited to 1,300 pixels. Measurements are planar estimates from visible pixels. Pressure witness color is a relative image signal and cannot establish force. Verify clearances on the physical tray before using them for a real object.

## Local launch

Run `./start.sh` and open `http://127.0.0.1:19991/`. Compose keeps inspections and image-preparation projects in its named volume. Run `./start.sh --down` to stop the application. Set `FOAMNEST_PORT` for a different local port.

For a native session, install the packages locked in `requirements.txt` and run `HOST=127.0.0.1 PORT=19991 python -m foamnest`. Set `FOAMNEST_DATA` for an independent data directory.
