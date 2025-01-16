import { useState } from "react";
import { Check, ChevronsUpDown } from "lucide-react"

import { cn } from "@/lib/utils"
import { Button } from "@/components/ui/button"
import {
  Command,
  CommandEmpty,
  CommandGroup,
  CommandInput,
  CommandItem,
  CommandList,
} from "@/components/ui/command"
import {
  Popover,
  PopoverContent,
  PopoverTrigger,
} from "@/components/ui/popover"
import { City } from "@/types/types"

interface CityProps {
    cities: City[];
    selectedCity: City | null;
    label: string;
    select_message: string;
    setCity: (city: City | null) => void;
}

export function CitySelector({ cities, selectedCity, setCity, label, select_message }: CityProps) {
  const [open, setOpen] = useState(false)

  return (
    <div>
        <p className="text-sm text-muted-foreground">{label}</p>
        <Popover open={open} onOpenChange={setOpen}>
        <PopoverTrigger asChild>
            <Button
            variant="outline"
            role="combobox"
            aria-expanded={open}
            className="w-[250px] justify-between"
            >
            {selectedCity
                ? cities.find((city) => city.name === selectedCity.name)?.name
                : select_message}
            <ChevronsUpDown className="ml-2 h-4 w-4 shrink-0 opacity-50" />
            </Button>
        </PopoverTrigger>
        <PopoverContent className="w-[250px] p-0">
            <Command>
            <CommandInput placeholder="Search for city..." />
            <CommandList>
                <CommandEmpty>No cities found.</CommandEmpty>
                <CommandGroup>
                {[...(selectedCity ? [selectedCity] : []), ...cities.filter(city => city.name !== selectedCity?.name)].map((city: City) => (
                    <CommandItem
                    key={city.name}
                    value={city.name}
                    onSelect={(currentValue) => {
                        if (currentValue === selectedCity?.name) {
                            setCity(null)
                        } else {
                            setCity(cities.find((c) => c.name === currentValue) || null)
                        }
                        setOpen(false)
                    }}
                    >
                    <Check
                        className={cn(
                        "mr-2 h-4 w-4",
                        selectedCity?.name === city.name ? "opacity-100" : "opacity-0"
                        )}
                    />
                    {city.name}
                    </CommandItem>
                ))}
                </CommandGroup>
            </CommandList>
            </Command>
        </PopoverContent>
        </Popover>
    </div>
  )
}
