import { Map } from './Map';
import { cn } from '@/lib/utils';
import { AxiosError } from 'axios';
import { Button } from './ui/button';
import { useEffect, useState } from 'react';
import { CitySelector } from './CitySelector';
import { Check, ChevronsUpDown, Loader2 } from 'lucide-react';
import { CriteriaWeightSelector } from './CriteriaWeightSelector';
import { getAlgorithms, getCities, optimizeRoute } from '@/utils/api';
import { Popover, PopoverContent, PopoverTrigger } from './ui/popover';
import { City, CriteriaRequest, PlanResponse, PlanRequest } from '@/types/types';
import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from './ui/card';
import { Command, CommandEmpty, CommandGroup, CommandInput, CommandItem, CommandList, CommandSeparator } from './ui/command';

export default function TravelPlanner() {
    const [cities, setCities] = useState<City[]>([]);
    const [algorithms, setAlgorithms] = useState<string[]>([]);
    const [error, setError] = useState<string | null>(null);

    const [selectedDestinations, setSelectedDestinations] = useState<City[]>([]);
    const [openDestinations, setOpenDestionations] = useState(false);

    const [selectedAlgorithm, setSelectedAlgorithm] = useState<string | null>(null)
    const [openAlgorithm, setOpenAlgorithm] = useState(false);

    const [startCity, setStartCity] = useState<City | null>(null);
    const [endCity, setEndCity] = useState<City | null>(null);

    const [criteriaRequest, setCriteriaRequest] = useState<CriteriaRequest>({
        distance: 0.33,
        duration: 0.33,
        cost: 0.34
    });
    
    const [planResponse, setPlanResponse] = useState<PlanResponse | null>(null);
    const [loading, setLoading] = useState(false);

    useEffect(() => {
        const fetchCities = async () => {
            try {
                const cities = await getCities();
                setCities(cities.sort((a, b) => a.name.localeCompare(b.name)));
            } catch {
                setError('Failed to fetch cities. Please try again.');
            }
        }

        const fetchAlgorithms = async () => {
            try {
                const algorithms = await getAlgorithms();
                setAlgorithms(algorithms);
            } catch {
                setError('Failed to fetch algorithms. Please try again.');
            }
        }

        fetchCities();
        fetchAlgorithms();
    }, []);

    useEffect(() => {
        if (startCity !== null && !selectedDestinations.includes(startCity)) {
            setSelectedDestinations([...selectedDestinations, startCity]);
        }

        if (endCity !== null && !selectedDestinations.includes(endCity)) {
            setSelectedDestinations([...selectedDestinations, endCity]);
        }
    }, [selectedDestinations, startCity, endCity]);

    useEffect(() => {
        setError(null);
    }, [selectedAlgorithm, startCity, selectedDestinations, endCity, criteriaRequest, planResponse]);

    return (
        <div className="container flex flex-col xl:flex-row justify-center gap-2">
            <Card className="xl:w-1/4 flex flex-col justify-center gap-4">
                <CardHeader>
                    <CardTitle>Travel Planner</CardTitle>
                    <CardDescription>Plan your optimal travel route</CardDescription>
                </CardHeader>

                <CardContent>
                    <p className="text-sm text-muted-foreground">Select algorithm (Required):</p>
                    <Popover open={openAlgorithm} onOpenChange={setOpenAlgorithm}>
                        <PopoverTrigger asChild>
                            <Button
                            variant="outline"
                            role="combobox"
                            aria-expanded={openAlgorithm}
                            className="w-[250px] justify-between"
                            >
                                {selectedAlgorithm ? selectedAlgorithm : 'Select algorithm...'}
                                <ChevronsUpDown className="ml-2 h-4 w-4 shrink-0 opacity-50" />
                            </Button>
                        </PopoverTrigger>
                        <PopoverContent className="w-[250px] p-0">
                            <Command>
                                <CommandInput placeholder="Select algorithm..." />
                                <CommandList>
                                    <CommandEmpty>No algorithms found.</CommandEmpty>
                                    <CommandGroup>
                                        {algorithms.map((algorithm: string) => (
                                            <CommandItem
                                                key={algorithm}
                                                value={algorithm}
                                                onSelect={(currentValue) => {
                                                    setSelectedAlgorithm(currentValue);
                                                    setOpenAlgorithm(false);
                                                }}
                                            >
                                                <Check
                                                    className={cn(
                                                    "mr-2 h-4 w-4",
                                                    selectedAlgorithm === algorithm ? "opacity-100" : "opacity-0"
                                                    )}
                                                />
                                                {algorithm}
                                            </CommandItem>
                                        ))}
                                    </CommandGroup>
                                </CommandList>
                            </Command>
                        </PopoverContent>
                    </Popover>
                </CardContent>

                <CardContent>
                    <CitySelector cities={cities} selectedCity={startCity} setCity={setStartCity} label="Start city (Required, green):" select_message="Select a starting city..." />
                </CardContent>

                <CardContent>
                    <p className="text-sm text-muted-foreground">Select destination cities (Optional):</p>
                    <Popover open={openDestinations} onOpenChange={setOpenDestionations}>
                        <PopoverTrigger asChild>
                            <Button
                            variant="outline"
                            role="combobox"
                            aria-expanded={openDestinations}
                            className="w-[250px] justify-between"
                            >
                                {selectedDestinations.length > 0 
                                    ? `${selectedDestinations.length} destination ${selectedDestinations.length === 1 ? 'city' : 'cities'} selected`
                                    : 'Select destination cities...'}
                                <ChevronsUpDown className="ml-2 h-4 w-4 shrink-0 opacity-50" />
                            </Button>
                        </PopoverTrigger>
                        <PopoverContent className="w-[250px] p-0">
                            <Command>
                                <CommandInput placeholder="Select destination cities..." />
                                <CommandList>
                                    <CommandEmpty>No cities found.</CommandEmpty>
                                    <CommandGroup heading="Selected destinations">
                                        {selectedDestinations.map((city: City) => (
                                            <CommandItem
                                                key={city.name}
                                                value={city.name}
                                                onSelect={(currentValue) => {
                                                    if (currentValue === startCity?.name) {
                                                        setStartCity(null);
                                                    }

                                                    if (currentValue === endCity?.name) {
                                                        setEndCity(null);
                                                    }

                                                    setSelectedDestinations(selectedDestinations.filter(city => city.name !== currentValue));
                                                }}
                                            >
                                                <Check className="ml-2 h-4 w-4 shrink-0 opacity-50" />
                                                {city.name}
                                            </CommandItem>
                                        ))}
                                    </CommandGroup>
                                    <CommandSeparator />
                                    <CommandGroup heading="Available destinations">
                                        {cities.filter(city => !selectedDestinations.includes(city)).map((city: City) => (
                                            <CommandItem
                                                key={city.name}
                                                value={city.name}
                                                onSelect={(currentValue) => {
                                                    setSelectedDestinations([...selectedDestinations, cities.find(city => city.name === currentValue)!]);
                                                }}
                                            >
                                                {city.name}
                                            </CommandItem>
                                        ))}
                                    </CommandGroup>
                                </CommandList>
                            </Command>
                        </PopoverContent>
                    </Popover>
                </CardContent>

                <CardContent>
                    <CitySelector cities={cities} selectedCity={endCity} setCity={setEndCity} label="End city (Optional):" select_message="Select an ending city..." />
                </CardContent>

                <CardContent>
                    <p className="text-sm text-muted-foreground">Set the priority of criteria</p>
                    <CriteriaWeightSelector criteriaRequest={criteriaRequest} onWeightChange={setCriteriaRequest} />
                </CardContent>

                <CardFooter className='gap-4'>
                    <Button
                        className='w-[150px]'
                        disabled={selectedAlgorithm === null || startCity === null || loading}
                        onClick={async () => {
                            if (selectedDestinations.filter(city => city !== startCity && city !== endCity).length === 0) {
                                setSelectedDestinations(cities);
                            }

                            const planRequest: PlanRequest = {
                                algorithm: selectedAlgorithm!,
                                start: startCity!.name,
                                end: endCity?.name,
                                destinations: selectedDestinations.map(city => city.name),
                                criteria: criteriaRequest
                            };

                            try {
                                setLoading(true);
                                const planResponse = await optimizeRoute(planRequest);
                                setPlanResponse(planResponse);
                            } catch (error) {
                                setError((error as AxiosError<{detail: string}>).response?.data?.detail || 'Failed to optimize route. Please try again.');
                            }

                            setLoading(false);
                        }}
                    >
                        {loading && <Loader2 className="animate-spin mr-2 h-4 w-4" />}
                        Plan route
                    </Button>
                    <Button
                        variant="outline"
                        disabled={planResponse === null}
                        onClick={() => {
                            setPlanResponse(null);
                            setSelectedDestinations([]);
                            setSelectedAlgorithm(null);
                            setStartCity(null);
                            setEndCity(null);
                            setCriteriaRequest({
                                distance: 0.33,
                                duration: 0.33,
                                cost: 0.34
                            });
                        }}
                    >
                        Reset
                    </Button>
                </CardFooter>                

                {error && (
                    <CardContent>
                        <p className="text-sm text-red-500">{error}</p>
                    </CardContent>
                )}

                {planResponse && (
                    <CardContent>
                        <p className="text-sm text-muted-foreground">Optimal route:</p>
                        <p className="text-sm text-muted-foreground">Travel distance: {Math.round(planResponse.distance * 100) / 100} km</p>
                        <p className="text-sm text-muted-foreground">Travel duration: {Math.round(planResponse.duration * 100) / 100} hours</p>
                        <p className="text-sm text-muted-foreground">Travel cost: {Math.round(planResponse.cost * 100) / 100} €</p>
                    </CardContent>
                )}
            </Card>
            <Card className="xl:w-3/4 pt-6 min-h-[90vh]">
                <CardContent>
                    <Map cities={cities} selectedDestinations={[...selectedDestinations, ...[startCity, endCity].filter(city => city !== null)]} planResponse={planResponse} startCity={startCity} />
                </CardContent>
            </Card>
        </div>
    )
}
